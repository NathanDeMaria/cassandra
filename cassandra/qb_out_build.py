"""Building the quarterback availability index.

Split from `cassandra.predictor.qb_out` the way `game_control_build` is split
from `game_control`: that module is on the replay path and reads a JSON file,
this one reads play-by-play parquet and lives with the fitting stack. The
artifact is the seam.

The rule, per team per game:

    starter = the first man to start for the team this season, until
              either a more established man starts in his place or a cover
              has started `HANDOVER[league]` games in a row without him --
              in a `KEEP_UNSEEN` league, only once the starter has been seen
              since, taking a snap in a game he didn't start
    out     = the starter took no pass, rush or sack

Established is more starts this season, or the same number and the earlier
first one. A team's first game of a season is what names the starter and is
never flagged -- the question "is he missing?" needs somebody to be missing.
Each flag carries which game of the absence it is (`runs`), which the index
weighs down game by game (`cassandra.predictor.qb_out.QB_OUT_DECAY`).

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

## A starter nobody has seen since

A one-game handover ends a college absence as soon as it starts, injury or
not, and an injured starter's team then plays on unflagged with a rating
still mostly his: replayed at the published `glicko_margin_units_success`
fit, ncaafb 2014-2026, the second and third games of an absence ran 2.6
points of win probability above the result (t ~3). So on ncaafb the cover
takes the job only once the starter has been seen healthy -- in relief, or
on the field in a game he didn't start, which is what a benching looks like
-- or by out-starting him. Until then the starter is out, every game, the
flag weighing less each one. Brier x1e-5 against handover 1, every flag
known: -4.9 +/- 2.9; priced as a live run would
(`cassandra.qb_starts.availability`): -8.2 +/- 2.5, FBS v FBS -17.1 +/- 8.8.
Dropping the out-starting limit was no better.

"Seen" is a rough stand-in for healthy. Of the starters it reads as benched,
68% started again that season, against 56% of the unseen ones -- the plays
know who played, not why -- so the rule leans on the decay, not the call.

Read `cassandra.predictor.qb_out` on the lookahead before reading any number
this produces. Whether a quarterback played is taken from the plays of the
game itself, which nobody has at kickoff.

The same walk keeps the latest season's quarterbacks game by game
(`build_all`, `cassandra.qb_starts`), for `picks.py` to ask whether a
team's starter is in doubt for its next game.
"""

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple, Protocol

from call_it_what_you_want import TeamNamer
from endgame.types import Season, iter_weeks
from say_youll_remember_me.qb import (
    TeamGameQb,
    ball_carrier,
    name_key,
    passer,
    team_games,
)

from cassandra.predictor.qb_out import QbOutFile, qb_out_path
from cassandra.qb_starts import (
    HANDOVER,
    KEEP_UNSEEN,
    GameQb,
    QbStartsFile,
    StarterWalk,
)

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


def registry_ids(
    namer: TeamNamer, home: str, away: str, offense_team_ids: Sequence[str | None]
) -> tuple[str, str] | None:
    """The home and away teams' ESPN ids from the registry, when both ran
    plays in this game; None otherwise.

    Exact where `home_away_ids` is a vote, and the vote can lose to the
    feed: ESPN's rows sometimes carry their scores out of order --
    Louisville and Stanford on 2026-10-31 read 17-17, then 52-24, then 17-17
    again down the plays -- and the swings outvoted the real scoring, which
    put each team's quarterback on the other and flagged both starters out.
    Five ncaafb games in 2026 read that way, and 32 and 53 in 2021 and 2022.
    The vote stays for what the registry can't answer: the NFL, which it
    doesn't cover, and a school it can't place.
    """
    ids = (namer.espn_id(home), namer.espn_id(away))
    present = {str(team_id) for team_id in offense_team_ids if team_id is not None}
    if ids[0] is None or ids[1] is None or ids[0] == ids[1]:
        return None
    if not {ids[0], ids[1]} <= present:
        return None
    return ids[0], ids[1]


def out_teams(
    ordered_games: Sequence[tuple[str, str]],
    parsed: Mapping[tuple[str, str], TeamGameQb],
    handover: int,
    keep_unseen: bool = False,
) -> dict[str, dict[str, int]]:
    """Which teams were missing their starter, by game id, and which game of
    his absence each was (1 for the first).

    `ordered_games` is (game_id, team) in the order that team played them,
    within one season. The caller does the ordering because it has the
    schedule and this has only the plays. `handover` is how many straight
    starts by one other man make him the starter instead; with `keep_unseen`
    they count only once the starter has been seen since (`KEEP_UNSEEN`).
    """
    missing: dict[str, dict[str, int]] = {}
    walks: dict[str, StarterWalk] = {}
    for game_id, team in ordered_games:
        current = parsed.get((game_id, team))
        if current is None:
            # No passing at all in this game -- a team that never threw
            # tells us nothing about who was available, and calling that
            # "out" would flag every wildcat afternoon as an injury.
            continue
        walk = walks.setdefault(team, StarterWalk(handover, keep_unseen))
        run = walk.step(current.starter_key, current.snap_keys)
        if run is not None:
            missing.setdefault(game_id, {})[team] = run
    return missing


def game_qbs(
    offense_team_ids: Sequence[str | None],
    texts: Sequence[str | None],
    periods: Sequence[int | None],
    home_scores: Sequence[int | None],
    away_scores: Sequence[int | None],
    home_id: str,
    parsed: Mapping[str, TeamGameQb],
) -> dict[str, GameQb]:
    """Per team id, one game's quarterbacks for `cassandra.qb_starts`.

    The plays in game order; `parsed` is `team_games` on them, by team id,
    and names the starter and who took a snap. This adds the passes each
    man threw and when he last took a snap -- the period, and his team's
    lead then -- which is what tells a starter hurt in a close game from
    one rested in a rout.
    """
    attempts: dict[str, Counter[str]] = {}
    names: dict[str, dict[str, str]] = {}
    last: dict[str, dict[str, tuple[int | None, int | None]]] = {}
    home = away = 0
    for team_id, text, period, home_score, away_score in zip(
        offense_team_ids, texts, periods, home_scores, away_scores
    ):
        if home_score is not None and away_score is not None:
            home, away = home_score, away_score
        if team_id is None:
            continue
        team_id = str(team_id)
        carrier = ball_carrier(text)
        if carrier is not None:
            lead = home - away if team_id == home_id else away - home
            last.setdefault(team_id, {})[name_key(carrier)] = (period, lead)
        thrower = passer(text)
        if thrower is not None:
            key = name_key(thrower)
            attempts.setdefault(team_id, Counter())[key] += 1
            names.setdefault(team_id, {}).setdefault(key, thrower)
    return {
        team_id: GameQb(
            starter=qb.starter_key,
            attempts=dict(attempts.get(team_id, {})),
            names=names.get(team_id, {}),
            snaps=sorted(qb.snap_keys),
            last_snap=last.get(team_id, {}),
        )
        for team_id, qb in parsed.items()
    }


class Built(NamedTuple):
    """What one walk of the plays finds."""

    #: game id -> the teams missing their starter (`cassandra.predictor.qb_out`).
    missing: dict[str, set[str]]
    #: The latest season's quarterbacks, game by game (`cassandra.qb_starts`);
    #: None with no seasons.
    starts: QbStartsFile | None
    #: game id -> team -> which game of his absence each flag was.
    runs: dict[str, dict[str, int]] = {}


async def build(
    league: str,
    seasons: Sequence[Season],
    source: PlayWeeks,
    namer: TeamNamer | None = None,
) -> dict[str, set[str]]:
    """Who was missing, by game id: `build_all`'s index alone."""
    return (await build_all(league, seasons, source, namer)).missing


async def build_all(
    league: str,
    seasons: Sequence[Season],
    source: PlayWeeks,
    namer: TeamNamer | None = None,
) -> Built:
    """Walk every season's plays and work out who was missing.

    Rebuilt whole rather than topped up. Unlike the control index there is no
    fit behind this to key idempotency on, and the parse is cheap next to the
    read -- so "what does the current parser say about every game" is the
    only question worth answering, and merging two parsers' opinions into one
    file is the thing to avoid.

    A play's `offense_team_id` is turned into a team name per game through
    the team registry where it can place both teams (`registry_ids`), and by
    `home_away_ids`' vote where it can't: the registry is an NCAA one and
    knows nothing about the NFL. `namer` is applied to the games first, so
    the names this writes are the ones a `Matchup` carries.
    """
    namer = namer if namer is not None else TeamNamer.for_league(league)
    missing: dict[str, set[str]] = {}
    runs: dict[str, dict[str, int]] = {}
    latest = max((s.year for s in seasons), default=None)
    starts: dict[str, dict[str, GameQb]] = {}
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
                home_id, away_id = registry_ids(
                    namer, names[0], names[1], offense
                ) or home_away_ids(
                    offense,
                    [columns["home_score"][i] for i in rows],
                    [columns["away_score"][i] for i in rows],
                )
                if home_id is None or away_id is None:
                    continue
                by_id = {home_id: names[0], away_id: names[1]}
                texts = [columns["text"][i] for i in rows]
                found = {
                    team_id: value
                    for (_, team_id), value in team_games(
                        [game_id] * len(rows), offense, texts
                    ).items()
                }
                for team_id, value in found.items():
                    team = by_id.get(team_id)
                    if team is not None:
                        parsed[(game_id, team)] = value
                if season.year == latest:
                    qbs = game_qbs(
                        offense,
                        texts,
                        [columns["period"][i] for i in rows],
                        [columns["home_score"][i] for i in rows],
                        [columns["away_score"][i] for i in rows],
                        home_id,
                        found,
                    )
                    starts[game_id] = {
                        by_id[team_id]: qb
                        for team_id, qb in qbs.items()
                        if team_id in by_id
                    }

        ordered: dict[str, list[tuple[str, str]]] = {}
        for week in weeks:
            for game in week.games_in_order:
                if not game.completed:
                    continue
                for team in (game.home, game.away):
                    ordered.setdefault(team, []).append((game.game_id, team))
        for team_games_in_order in ordered.values():
            for game_id, teams in out_teams(
                team_games_in_order,
                parsed,
                HANDOVER[league],
                keep_unseen=league in KEEP_UNSEEN,
            ).items():
                missing.setdefault(game_id, set()).update(teams)
                runs.setdefault(game_id, {}).update(teams)
    return Built(
        missing,
        None
        if latest is None
        else QbStartsFile(league=league, season=latest, games=starts),
        runs,
    )


def with_season[V](
    stored: Mapping[str, V],
    season: Season,
    fresh: Mapping[str, V],
) -> dict[str, V]:
    """The stored index with one season's games replaced by a fresh build of
    that season alone: its flags, or its runs.

    Exact, not an approximation: the rule never looks past a season's own
    games (a team's first game names its starter), so building one season
    gives its games the flags a whole rebuild would. Every game of the
    season is dropped from the stored copy first, so a game the fresh build
    no longer flags -- a corrected box score, a starter re-read -- leaves
    rather than lingering.

    This is the one place two builds share a file, which the whole rebuild
    exists to avoid. The stored copy is the daily publish's whole rebuild,
    by the same image unless one shipped since, and the next daily rebuild
    puts the file back on one parser either way.
    """
    ids = {game.game_id for week in season.weeks for game in week.games}
    merged = {game: teams for game, teams in stored.items() if game not in ids}
    merged.update(fresh)
    return merged


def write(
    league: str,
    games: Mapping[str, Iterable[str]],
    runs: Mapping[str, Mapping[str, int]] | None = None,
    fixtures: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    """Save the index. Key-sorted, and team lists sorted, so a rebuild diffs."""
    path = qb_out_path(league)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = QbOutFile(
        league=league,
        games={game: sorted(teams) for game, teams in games.items()},
        runs={game: dict(teams) for game, teams in (runs or {}).items()},
        fixtures={game: dict(teams) for game, teams in (fixtures or {}).items()},
    )
    path.write_text(json.dumps(document.model_dump(mode="json"), sort_keys=True))
