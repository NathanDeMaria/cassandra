"""Building the quarterback availability index.

Split from `cassandra.predictor.qb_out` the way `game_control_build` is split
from `game_control`: that module is on the replay path and reads a JSON file,
this one reads play-by-play parquet and lives with the fitting stack. The
artifact is the seam.

The rule, per team per game:

    expected starter = whoever started that team's previous game this season
    out              = the expected starter took no pass, rush or sack

A team's first game of a season has no expected starter and is skipped -- the
question "is he missing?" needs somebody to be missing.

Read `cassandra.predictor.qb_out` on the lookahead before reading any number
this produces. Whether a quarterback played is taken from the plays of the
game itself, which nobody has at kickoff.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from call_it_what_you_want import TeamNamer
from endgame.types import Season, iter_weeks

from cassandra.predictor.qb_out import QbOutFile, qb_out_path
from cassandra.qb import TeamGameQb, team_games

#: Past any real season. NCAAFB runs to about 17 source weeks plus bowls,
#: the NFL to 22; this is the loop bound for "ask for everything".
_MAX_WEEK = 30


class PlayWeeks(Protocol):
    """What this build needs from a play store, so a test can hand it a dict.

    A protocol rather than a base class: the real implementation is
    `endgame_aws`'s store wrapped to return an empty table for a week that
    isn't there, and a test hands over something backed by a dict.
    """

    async def load_week(self, league: str, season: int, week: int) -> Any: ...


def home_away_ids(
    offense_team_ids: Sequence[str | None],
    home_scores: Sequence[int | None],
    away_scores: Sequence[int | None],
) -> tuple[str | None, str | None]:
    """Which of a game's two team ids is the home side, read off the scoring.

    Voted rather than looked up. The obvious way to turn a play's
    `offense_team_id` into a team name is a registry, and
    `call_it_what_you_want` is an NCAA one -- `TeamNamer.for_league("nfl")`
    is empty, so every id resolved to None and the first nfl index built
    came out with zero games in it across forty-seven seasons.

    So it is derived from the plays instead, which works for any league with
    play-by-play and cannot disagree with the names the replay uses. Each
    play where the home score went up is a vote that its offense is the home
    team, and each play where the away score went up is a vote against.
    Majority wins, which absorbs the plays where the scoring side isn't the
    offense -- a pick six, a punt return -- without needing to know which
    those were.

    (None, None) for a game with no scoring at all to vote on, or one where
    the vote is tied, which is a game this cannot place and had better skip.
    """
    # Every team that ran a play starts at zero, so a side that never scored
    # is still rankable: one team's negative votes place the other, and a
    # shutout would otherwise leave only one candidate and no game.
    votes: Counter[str] = Counter(
        {str(team_id): 0 for team_id in offense_team_ids if team_id is not None}
    )
    previous: tuple[int, int] | None = None
    for team_id, home, away in zip(offense_team_ids, home_scores, away_scores):
        if home is None or away is None:
            continue
        current = (home, away)
        if previous is not None and team_id is not None:
            votes[str(team_id)] += (current[0] - previous[0]) - (
                current[1] - previous[1]
            )
        previous = current
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < 2 or ranked[0][1] <= ranked[-1][1]:
        return (None, None)
    return (ranked[0][0], ranked[-1][0])


def out_teams(
    ordered_games: Sequence[tuple[str, str]],
    parsed: Mapping[tuple[str, str], TeamGameQb],
) -> dict[str, set[str]]:
    """Which teams were missing their expected starter, by game id.

    `ordered_games` is (game_id, team) in the order that team played them,
    within one season. The caller does the ordering because it has the
    schedule and this has only the plays.
    """
    missing: dict[str, set[str]] = {}
    previous: dict[str, TeamGameQb] = {}
    for game_id, team in ordered_games:
        current = parsed.get((game_id, team))
        expected = previous.get(team)
        if current is not None:
            previous[team] = current
        if expected is None or current is None:
            # No expected starter yet, or no passing at all in this game --
            # a team that never threw tells us nothing about who was
            # available, and calling that "out" would flag every wildcat
            # afternoon as an injury.
            continue
        if expected.starter_key not in current.snap_keys:
            missing.setdefault(game_id, set()).add(team)
    return missing


async def build(
    league: str,
    seasons: Sequence[Season],
    source: PlayWeeks,
    namer: TeamNamer | None = None,
) -> dict[str, set[str]]:
    """Walk every season's plays and work out who was missing.

    Rebuilt whole rather than topped up. Unlike the control index there is no
    fit behind this to key idempotency on, and the parse is cheap next to the
    read -- so "what does the current parser say about every game" is the
    only question worth answering, and merging two parsers' opinions into one
    file is the thing to avoid.

    A play's `offense_team_id` is turned into a team name per game, by
    `home_away_ids`, rather than through the team registry: the registry is
    an NCAA one and knows nothing about the NFL. `namer` is still applied to
    the games first, so the names this writes are the ones a `Matchup`
    carries.
    """
    namer = namer if namer is not None else TeamNamer.for_league(league)
    missing: dict[str, set[str]] = {}
    for season in sorted(seasons, key=lambda s: s.year):
        # `namer.apply` first, exactly as `generate_predictions` does before
        # the predictor sees a game. The index is looked up by the name in a
        # `Matchup`, which is the canonical one, so an index keyed by the raw
        # name would silently miss every school that has been renamed.
        weeks = [
            week._replace(games=[namer.apply(game) for game in week.games])
            for week in iter_weeks(season)
        ]
        sides = {
            game.game_id: (game.home, game.away)
            for week in weeks
            for game in week.games_in_order
        }
        parsed: dict[tuple[str, str], TeamGameQb] = {}
        # A fixed range rather than the week numbers `iter_weeks` hands back.
        # The play store is keyed by the *source's* week numbering and
        # `iter_weeks` rebuilds calendar weeks for a league with a
        # `season_start`, so the two need not agree -- ncaafb 2026's calendar
        # weeks start at 2 while its plays are filed under 2, 3 and 4. Asking
        # for every plausible number and merging what comes back sidesteps the
        # question: the result is keyed by (game id, team), so a week that
        # isn't there contributes nothing and one asked for twice cannot
        # double count.
        for number in range(1, _MAX_WEEK + 1):
            table = await source.load_week(league, season.year, number)
            if table is None or table.num_rows == 0:
                continue
            columns = table.to_pydict()
            by_game: dict[str, list[int]] = {}
            for row, game_id in enumerate(columns["game_id"]):
                by_game.setdefault(str(game_id), []).append(row)
            for game_id, rows in by_game.items():
                names = sides.get(game_id)
                if names is None:
                    continue
                offense = [columns["offense_team_id"][i] for i in rows]
                home_id, away_id = home_away_ids(
                    offense,
                    [columns["home_score"][i] for i in rows],
                    [columns["away_score"][i] for i in rows],
                )
                if home_id is None or away_id is None:
                    continue
                by_id = {home_id: names[0], away_id: names[1]}
                for (_, team_id), value in team_games(
                    [game_id] * len(rows), offense, [columns["text"][i] for i in rows]
                ).items():
                    team = by_id.get(team_id)
                    if team is not None:
                        parsed[(game_id, team)] = value

        ordered: dict[str, list[tuple[str, str]]] = {}
        for week in weeks:
            for game in week.games_in_order:
                if not game.completed:
                    continue
                for team in (game.home, game.away):
                    ordered.setdefault(team, []).append((game.game_id, team))
        for team_games_in_order in ordered.values():
            for game_id, teams in out_teams(team_games_in_order, parsed).items():
                missing.setdefault(game_id, set()).update(teams)
    return missing


def write(league: str, games: Mapping[str, set[str]]) -> None:
    """Save the index. Key-sorted, and team lists sorted, so a rebuild diffs."""
    path = qb_out_path(league)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = QbOutFile(
        league=league, games={game: sorted(teams) for game, teams in games.items()}
    )
    path.write_text(json.dumps(document.model_dump(mode="json"), sort_keys=True))
