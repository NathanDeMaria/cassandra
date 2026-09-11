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
from collections.abc import Mapping, Sequence

from call_it_what_you_want import TeamNamer
from endgame.types import Season, iter_weeks

from cassandra.predictor.qb_out import QbOutFile, qb_out_path
from cassandra.qb import TeamGameQb, team_games

#: Past any real season. NCAAFB runs to about 17 source weeks plus bowls,
#: the NFL to 22; this is the loop bound for "ask for everything".
_MAX_WEEK = 30


class PlayWeeks:
    """What this build needs from a play store, so a test can hand it a dict."""

    async def load_week(self, league: str, season: int, week: int): ...


def _canonical(namer: TeamNamer, teams: Sequence[str]) -> dict[str, str]:
    """ESPN team id -> canonical name, for the teams a league actually rates.

    Built from the names the replay uses rather than from the registry at
    large, so the ids that resolve are exactly the ones a matchup will be
    keyed by. A team whose name doesn't resolve to an id is dropped: it can
    never be looked up, and an entry nothing matches is worse than none.
    """
    found: dict[str, str] = {}
    for team in teams:
        espn_id = namer.espn_id(team)
        if espn_id is not None:
            found[str(espn_id)] = team
    return found


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
        names = _canonical(
            namer,
            sorted(
                {
                    team
                    for week in weeks
                    for game in week.games
                    for team in (game.home, game.away)
                }
            ),
        )
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
            for (game_id, team_id), value in team_games(
                columns["game_id"], columns["offense_team_id"], columns["text"]
            ).items():
                team = names.get(team_id)
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
