"""Walking a league's play-by-play, for whichever number you take off a game.

Two indexes are built this way now -- `cassandra.game_control_build` takes the
share of the game a team controlled, `cassandra.epa_build` takes what its
offense added per snap -- and everything except that one line is the same
work: the same weekly parquet, the same grouping into games, the same
questions about whether the plays can be trusted to describe the game the
schedule recorded. This is that shared half, and a build module is the
scoring line plus a file format.

What a sweep does, a week at a time: read the week's processed plays, group
them into games, ask `score` for the one small thing worth keeping, and drop
the plays. Peak memory is a handful of weeks rather than a league's history --
ncaafb is about seven million plays and sixteen thousand answers, and only the
second number is worth holding.

Weeks are the unit because they are how the plays are stored: one parquet
object per league-week, which is also the granularity `iter_weeks` hands out,
so the sweep's `(year, week)` pairs are the same ones the raw pull wrote
under. Nothing here has to guess a key or list a prefix.

What the first full control sweep found, which is the calibration for the
filters here: the NFL keeps 4,984 of the 4,989 games it has plays for, and
ncaafb 25,307 of 26,330 (96%). Against the whole schedule those are smaller
shares -- cassandra replays football from 1980 and the processed plays start
in 2006, and only about a third of an NCAAFB week is a game ESPN covers at
all. Those filters are about the plays rather than about the number, so a
second index built through here inherits the same coverage.

In the `fit` group with the rest of the fitting stack, not on the replay path:
this imports `lucky_ones.arrow` and pyarrow and moves ~190MB of parquet, while
what a predictor reads is the small JSON file at the other end. The artifact
is the seam.
"""

import asyncio
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, distribution, version
from logging import getLogger
from typing import NamedTuple

from endgame.types import Game, Season, iter_weeks
from lucky_ones import GamePlays, group_by_game
from lucky_ones.plays import Play, PlaySource
from pydantic import BaseModel

logger = getLogger(__name__)

# How many weeks are in flight at once. The work is a read of a ~0.5MB object
# followed by a matrix multiply, so it is latency-bound and wants some
# concurrency -- but every week in flight is holding its plays, and ncaafb's
# are ~21,000 to a week, so the whole season at once is the wrong end of the
# trade.
MAX_CONCURRENT_WEEKS = 8

# Coverage is checked against the score, not against the clock.
#
# `GameControl.seconds` looks like a coverage measure and is not one: each
# snap is weighted by the clock from it to the *next* snap, and the last snap
# is charged with everything left in regulation. So a game whose play-by-play
# stops at halftime reports the same 3600 as one that runs to the whistle --
# measured, not assumed. What separates them is whether the plays ever reach
# the score the game finished at, which is what `covers` asks. EPA's play
# counts have the same shape of problem and the same answer.

type GameScorer[T] = Callable[[GamePlays], T | None]
"""What a build module contributes: one game in, one small thing out.

None means the game came through every check here and still had nothing to
give -- no elapsed regulation clock to average a curve over, no snaps to
average an EPA across -- which is `SweepStats.unscored` and is the one drop
reason the sweep itself can't diagnose.
"""


def lucky_ones_revision() -> str:
    """The installed `lucky-ones`, as specifically as the install can say.

    A git install records its commit under PEP 610, which is the identity
    worth having: the package version is `0.1.0` and has been since it was
    written, while the code that turns plays into a curve moves. Falling back
    to the version rather than raising, because an install that isn't from
    git -- a wheel, an editable checkout -- is a development environment, and
    failing the sweep there would be worse than comparing something coarser.
    """
    try:
        raw = distribution("lucky-ones").read_text("direct_url.json")
    except PackageNotFoundError:
        raw = None
    if raw:
        vcs = json.loads(raw).get("vcs_info") or {}
        commit = vcs.get("commit_id")
        if commit:
            return commit
    return version("lucky-ones")


def fit_differences(stored: BaseModel, current: BaseModel) -> str:
    """Which fields of a fit header moved, for the line a full sweep prints.

    Named rather than counted, and the win probability `run_id` on its own
    would have been the wrong thing to print: the rev that added the
    luck-adjusted reading left the fits inside the package byte-identical, so
    it moves `lucky_ones` and `reading` while the run id stays exactly where
    it was.

    Takes any two pydantic models of the same type -- `ControlFit` and
    `EpaFit` are different shapes and neither of them is this module's
    business, so what it needs is `model_fields` and equality.
    """
    return ", ".join(
        name
        for name in type(stored).model_fields
        if getattr(stored, name) != getattr(current, name)
    )


class SweepStats(NamedTuple):
    """What a sweep did, as counts of games, for the line the job prints.

    Every drop reason is separate because the first real run is the only
    thing that can say whether the filters are set right, and "n games were
    dropped" wouldn't tell anyone which one to loosen.
    """

    scored: int
    """Games that came out with a usable number."""

    incomplete: int
    """Dropped: the plays don't reach the score the game finished at."""

    unmatched: int
    """Dropped: plays for a game that isn't a completed game of this season.

    Ordinary rather than alarming -- a game that was still in progress when
    the plays were pulled looks like this -- and it costs nothing, since a
    game the schedule doesn't have is a game no replay will ask about.
    """

    no_score: int
    """Dropped: no play in the game carries a score, so there is no final."""

    unscored: int
    """Dropped: the scorer had nothing to give for a game that passed
    everything else.

    Which is a different sentence per index, and neither of them is a
    complaint about the plays: for game control it is a game with no elapsed
    regulation clock to average a curve over, and for EPA a game with no
    regulation snap for one of the two offenses.
    """

    played: int
    """Completed games in the seasons swept, whether or not they had plays."""

    disagreed: int
    """Dropped: the plays and the stored result name different winners.

    Not an orientation failure, which is what it was put here to catch.
    `lucky_ones` orients a curve by the play table's own
    `home_score`/`away_score` -- ESPN's labelling, the same source
    `Game.home` comes from -- so the sides agree by construction, and the
    first full sweep bears that out: zero across the NFL's 4,989 games with
    play-by-play.

    ncaafb reports 136 of 25,443 (0.5%), and their shape is what says this
    is data rather than a bug. They are spread evenly over twenty seasons
    rather than concentrated in one, and only 9 are a clean home/away swap;
    the rest are play-by-play whose last scoring play simply doesn't match
    the game's stored result. Those games keep their real score, which is
    what every game without plays already does -- so the conservative
    reading costs nothing and a corrective flip would have been guessing at
    126 games it wouldn't have fixed.
    """

    @property
    def with_plays(self) -> int:
        """Scheduled games the play store had anything for, kept or not.

        The denominator that says whether the filters are set right. The
        other one -- every completed game in every season swept -- is mostly
        answering a different question, since cassandra replays the NFL from
        1980 and the processed plays start in 2006.
        """
        return (
            self.scored + self.incomplete + self.no_score + self.unscored + self.disagreed
        )

    def __str__(self) -> str:
        def share(of: int) -> str:
            return f"{self.scored / of:.0%}" if of else "n/a"

        dropped = ", ".join(
            f"{count} {label}"
            for label, count in (
                ("incomplete", self.incomplete),
                ("unmatched", self.unmatched),
                ("no score", self.no_score),
                ("unscored", self.unscored),
            )
            if count
        )
        line = (
            f"{self.scored} of {self.with_plays} games with plays "
            f"({share(self.with_plays)}), {self.played} played "
            f"({share(self.played)} of the schedule)"
        )
        if dropped:
            line += f" -- dropped {dropped}"
        if self.disagreed:
            line += f" -- {self.disagreed} DISAGREE WITH THE STORED RESULT"
        return line


def add(left: SweepStats, right: SweepStats) -> SweepStats:
    return SweepStats(*(a + b for a, b in zip(left, right)))


NOTHING = SweepStats(
    scored=0,
    incomplete=0,
    unmatched=0,
    no_score=0,
    unscored=0,
    played=0,
    disagreed=0,
)


def weeks_in(seasons: Iterable[Season]) -> list[tuple[int, int]]:
    """The `(year, week)` pairs to sweep, in the order a season runs.

    `iter_weeks` rather than `season.weeks` because it is what the play pull
    walked when it chose the week numbers in the keys -- ESPN's own for the
    NFL, calendar weeks from the start of the season for NCAAFB. Reading them
    any other way would ask the store for a week that isn't there.
    """
    return [
        (season.year, week.number)
        for season in sorted(seasons, key=lambda s: s.year)
        for week in iter_weeks(season)
    ]


def completed_games(seasons: Iterable[Season]) -> dict[str, Game]:
    """Every completed game in `seasons`, by id.

    The sweep's reference for three questions, none of them orientation:
    whether a game with plays is one the schedule has (`unmatched`), whether
    its plays reach the score it finished at (`covers`), and whether the two
    sources name the same winner (`disagrees`).
    """
    return {
        game.game_id: game
        for season in seasons
        for week in season.weeks
        for game in week.games
        if game.completed
    }


def final_from_plays(plays: Sequence[Play]) -> tuple[int, int] | None:
    """The score the plays end at, as (home, away).

    The last play that carries one, not the maximum. A cumulative score
    looks like it can only go up, and it can go down: a touchdown reversed
    on review or wiped out by a penalty is scored on its own play and taken
    off again on a later one. Reading the maximum picks up the points that
    were taken back, which shows up as a game the plays and the schedule
    disagree about -- five of them across the NFL's twenty seasons, four
    with the plays exactly six points ahead of the result.

    Scanned from the end rather than taken off `plays[-1]`, because the last
    row of a game is usually an administrative one (END GAME) whose columns
    can be null. Same reading as `lucky_ones.state.final_outcome`, which is
    the other thing in this pipeline that has to answer the question.

    None for a game where no play carries a score at all, which the play
    table does contain.
    """
    for play in reversed(plays):
        if play.home_score is not None and play.away_score is not None:
            return play.home_score, play.away_score
    return None


def disagrees(final: tuple[int, int], played: Game) -> bool:
    """Do the plays and the stored result disagree about who *won*?

    A stronger contradiction than `covers`. Play-by-play that stops early
    falls short of the final score without contradicting it; this is the
    case where the two sources name different winners, which orientation
    alone should not be able to produce -- see `SweepStats.disagreed` for
    what the first sweep found instead.

    Compared by sign, and only when both sides are decisive, so a tie on
    either end isn't read as a contradiction.
    """
    from_plays = final[0] - final[1]
    from_result = played.home_score - played.away_score
    if not from_plays or not from_result:
        return False
    return (from_plays > 0) != (from_result > 0)


def covers(final: tuple[int, int], played: Game) -> bool:
    """Do the plays run all the way to the game's final score?

    The coverage test, and it is an equality rather than a tolerance on
    purpose. A game the plays only half cover still produces a perfectly
    well-formed number, and that number describes a different game -- one
    where somebody led at halftime -- so the safe answer to "did the
    play-by-play cover this" is the strict one. A game that fails here keeps
    its real score, which is what every game without plays already does.

    If this turns out to reject games it shouldn't, the sweep's own report is
    where that shows up: `incomplete` counts them, and it is printed next to
    the games that came through.
    """
    return final == (played.home_score, played.away_score)


async def _sweep_week[T](
    league: str,
    source: PlaySource,
    year: int,
    week: int,
    played: Mapping[str, Game],
    score: GameScorer[T],
) -> tuple[dict[str, T], SweepStats]:
    """One week's plays, reduced to one small thing per game.

    The plays go out of scope with this call, which is the point of doing it
    per week rather than assembling a season and scoring it.
    """
    plays = await source.load_week(league, year, week)
    if not plays:
        return {}, NOTHING

    scored: dict[str, T] = {}
    incomplete = unmatched = no_score = unscored = disagreed = 0
    for game in group_by_game(plays):
        result = played.get(game.game_id)
        if result is None:
            unmatched += 1
            continue
        final = final_from_plays(game.plays)
        if final is None:
            no_score += 1
            continue
        if disagrees(final, result):
            disagreed += 1
            logger.warning(
                "%s %d week %d: %s ended %d-%d in the plays and %d-%d in the "
                "schedule; not scoring it",
                league,
                year,
                week,
                game.game_id,
                *final,
                result.home_score,
                result.away_score,
            )
            continue
        if not covers(final, result):
            incomplete += 1
            continue
        value = score(game)
        if value is None:
            unscored += 1
            continue
        scored[game.game_id] = value

    return scored, NOTHING._replace(
        scored=len(scored),
        incomplete=incomplete,
        unmatched=unmatched,
        no_score=no_score,
        unscored=unscored,
        disagreed=disagreed,
    )


async def sweep[T](
    league: str,
    seasons: Sequence[Season],
    source: PlaySource,
    score: GameScorer[T],
) -> tuple[dict[str, T], SweepStats]:
    """`score` applied to every game of `seasons` whose play-by-play covers it.

    Weeks run concurrently up to `MAX_CONCURRENT_WEEKS`; a week nobody has
    processed comes back empty rather than raising, which is the normal state
    of the rest of a season in September.
    """
    played = completed_games(seasons)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEEKS)

    async def one(year: int, week: int):
        async with semaphore:
            return await _sweep_week(league, source, year, week, played, score)

    results = await asyncio.gather(
        *(one(year, week) for year, week in weeks_in(seasons))
    )

    scored: dict[str, T] = {}
    stats = NOTHING._replace(played=len(played))
    for week_scored, week_stats in results:
        scored.update(week_scored)
        stats = add(stats, week_stats)
    return scored, stats._replace(scored=len(scored))
