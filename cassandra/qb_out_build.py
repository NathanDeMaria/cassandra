"""Building the quarterback availability index.

Split from `cassandra.predictor.qb_out` the way `game_control_build` is split
from `game_control`: that module is on the replay path and reads a JSON file,
this one reads play-by-play parquet and lives with the fitting stack. The
artifact is the seam.

The rule, per team per game:

    starter = the first man to start for the team this season, until
              either a more established man starts in his place or a cover
              has started `HANDOVER[league]` games in a row without him
    out     = the starter took no pass, rush or sack

Established is more starts this season, or the same number and the earlier
first one. A team's first game of a season is what names the starter and is
never flagged -- the question "is he missing?" needs somebody to be missing.

## Why the whole absence, and why the handover is per league

The first version of this was one game deep: the expected starter was
whoever started the previous game. The week a starter came back, the backup
who had covered was the expected man, took no snap, and the team was
flagged for getting its quarterback back. Runs of consecutive flags gave it
away -- 185 NFL runs of exactly two against 14 of three, a one-week absence
plus its return rather than two-week injuries. Keeping the senior man as the
starter through his absence fixed that, and also flags every game of it
rather than the first: the penalty said "a backup is starting today" for
one week and then nothing, while the man was still out. Replayed on the
NFL at the `glicko_full` fit, best brier over the penalty:

    one game deep, as shipped        0.220033
    first game of an absence only    0.219559
    whole absence, handover 2        0.219372
    whole absence, handover 3        0.219130
    whole absence, handover 4        0.219324
    whole absence, never             0.219387

College is a different sport for this. Its quarterback changes are mostly
not injuries -- a battle settled in week three, a benching, a platoon --
and from the second game of one of those on, the man playing is the team's
real quarterback and the team is *better*, not worse. Scored on the
`glicko_full` margin residual of every flagged team-game (positive means
the team beat the model's expectation), 2023-2025 ncaafb and 2015-2024 NFL:

                                   ncaafb                 nfl
    handover 1 (first game only)   n=822  -1.93 pts     n=332  -3.02 pts
    handover 3 (whole absence)     n=1394 -1.11         n=628  -2.73

Same mean on the NFL either way, so the extra games are worth flagging;
on ncaafb the extra 570 team-games run about +2.5, the sign of a switch
that improved the team. Hence the constant is per league.

`HANDOVER` counts consecutive starts by one man: shorter and a real absence
ends too early; longer and a starter lost for the season keeps his team
penalized for a backup who has, by then, become the team the rating
measures. Two backups alternating are a team without a starter, not a new
one.

The "more established" clause is what an early-season return needs. With
two games played and one start apiece, the opener's starter is the coach's
pick, and him starting again is not the other man's absence -- that one
was found by hand, Illinois State's Rittenhouse hurt at Oklahoma, out a
week, back in week three. Starts rather than attempts because attempts fail
the same way: a starter hurt early and back two weeks later has thrown
fewer passes than the man who covered.

What this does not catch is a demotion. A starter benched for the man
behind him reads as an absence until the handover -- the plays cannot tell
a benching from an injury -- which on the NFL is three weeks. Rare there,
and on ncaafb, where it is common, the handover is one game.

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

#: Consecutive starts by one man, without the starter, that make him the
#: starter -- per league, because a college quarterback change is usually
#: not an injury. See the module docstring for the measurements.
HANDOVER: Mapping[str, int] = {"nfl": 3, "ncaafb": 1}


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
    handover: int,
) -> dict[str, set[str]]:
    """Which teams were missing their starter, by game id.

    `ordered_games` is (game_id, team) in the order that team played them,
    within one season. The caller does the ordering because it has the
    schedule and this has only the plays. `handover` is how many straight
    starts by one other man make him the starter instead.
    """
    missing: dict[str, set[str]] = {}
    starter: dict[str, str] = {}
    # Per team, who has been starting in the starter's absence and for how
    # many games running. Cleared the moment the starter takes a snap.
    covering: dict[str, tuple[str, int]] = {}
    # Per team, starts this season by man and the order they first started
    # in, for "more established". Both count only games already played.
    starts: dict[str, Counter[str]] = {}
    first_start: dict[str, dict[str, int]] = {}
    for game_id, team in ordered_games:
        current = parsed.get((game_id, team))
        if current is None:
            # No passing at all in this game -- a team that never threw
            # tells us nothing about who was available, and calling that
            # "out" would flag every wildcat afternoon as an injury.
            continue
        today = current.starter_key
        tally = starts.setdefault(team, Counter())
        order = first_start.setdefault(team, {})
        expected = starter.get(team)

        def standing(key: str) -> tuple[int, int]:
            return (tally[key], -order.get(key, len(order)))

        if expected is None or standing(today) > standing(expected):
            # First game, or the established man starting in place of a
            # cover: he has the job, and nobody is missing.
            starter[team] = today
            covering.pop(team, None)
        elif expected in current.snap_keys:
            covering.pop(team, None)
        else:
            missing.setdefault(game_id, set()).add(team)
            cover, streak = covering.get(team, (None, 0))
            streak = streak + 1 if cover == today else 1
            if streak >= handover:
                starter[team] = today
                covering.pop(team, None)
            else:
                covering[team] = (today, streak)
        tally[today] += 1
        order.setdefault(today, len(order))
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
            # The starter is whoever threw early, so the plays have to be in
            # game order. They arrive that way today; sorting is cheap
            # insurance against a store that stops promising it.
            for rows in by_game.values():
                rows.sort(key=lambda i: columns["sequence_number"][i] or 0)
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
            for game_id, teams in out_teams(
                team_games_in_order, parsed, HANDOVER[league]
            ).items():
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
