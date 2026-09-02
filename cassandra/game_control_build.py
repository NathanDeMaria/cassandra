"""Building the game control index: the half that reads play-by-play.

Split from `cassandra.predictor.game_control` on purpose. That module is on
the replay path and reads a JSON file; this one imports `lucky_ones.arrow`
and pyarrow and moves ~190MB of parquet, and lives in the `fit` group with
the rest of the fitting stack. The artifact is the seam between them.

What the sweep does, a week at a time: read the week's processed plays, group
them into games, score each game's win probability curve, keep the one float
the curve averages to, and drop the plays. Peak memory is a handful of weeks
rather than a league's history -- ncaafb is about seven million plays and
sixteen thousand floats, and only the second number is worth holding.

Weeks are the unit because they are how the plays are stored: one parquet
object per league-week, which is also the granularity `iter_weeks` hands out,
so the sweep's `(year, week)` pairs are the same ones the raw pull wrote
under. Nothing here has to guess a key or list a prefix.

Idempotency is keyed on the fit, not on the file existing. A stage that finds
its own `ControlFit` already stored refreshes only the season still being
played -- history can't change while the model doesn't -- and one that finds a
different fit rebuilds the league from scratch, because merging numbers from
two different win probability models into one index would produce a rating
nobody could reproduce.
"""

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, distribution, version
from logging import getLogger
from typing import NamedTuple

from endgame.types import Game, Season, iter_weeks
from lucky_ones import MODELS, group_by_game
from lucky_ones.plays import Play, PlaySource

from cassandra.predictor.game_control import (
    ControlFit,
    GameControlFile,
    game_control_path,
    read_game_control_file,
)
from cassandra.predictor.types import GameControl

logger = getLogger(__name__)

# How many weeks are in flight at once. The work is a read of a ~0.5MB object
# followed by a matrix multiply, so it is latency-bound and wants some
# concurrency -- but every week in flight is holding its plays, and ncaafb's
# are ~21,000 to a week, so the whole season at once is the wrong end of the
# trade.
_MAX_CONCURRENT_WEEKS = 8

# Coverage is checked against the score, not against the clock.
#
# `GameControl.seconds` looks like a coverage measure and is not one: each
# snap is weighted by the clock from it to the *next* snap, and the last snap
# is charged with everything left in regulation. So a game whose play-by-play
# stops at halftime reports the same 3600 as one that runs to the whistle --
# measured, not assumed. What separates them is whether the plays ever reach
# the score the game finished at, which is what `_covers` asks.


def _lucky_ones_revision() -> str:
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


def current_fit(league: str) -> ControlFit:
    """What an index built right now would be built by.

    Raises `KeyError` for a league `lucky_ones` ships no fit for, which is
    every league that isn't football -- see `CONTROL_LEAGUES`.
    """
    return ControlFit(lucky_ones=_lucky_ones_revision(), run_id=MODELS[league].run_id)


class SweepStats(NamedTuple):
    """What a sweep did, as counts of games, for the line the job prints.

    Every drop reason is separate because the first real run is the only
    thing that can say whether the filters are set right, and "n games were
    dropped" wouldn't tell anyone which one to loosen.
    """

    scored: int
    """Games that came out with a usable control number."""

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

    no_clock: int
    """Dropped: no elapsed regulation time, so there is nothing to average."""

    played: int
    """Completed games in the seasons swept, whether or not they had plays."""

    disagreed: int
    """Dropped, and shouted about: the plays and the result disagree on the winner.

    Expected to be zero. `lucky_ones` orients a curve by the play table's own
    `home_score`/`away_score`, which is ESPN's labelling -- the same source
    `Game.home` comes from -- so the two agree by construction rather than by
    a join. This is the assertion that they do. A number above zero means one
    of them is not what this comment says it is, and a corrective flip before
    anyone has looked would be guessing.
    """

    def __str__(self) -> str:
        share = f"{self.scored / self.played:.0%}" if self.played else "n/a"
        dropped = ", ".join(
            f"{count} {label}"
            for label, count in (
                ("incomplete", self.incomplete),
                ("unmatched", self.unmatched),
                ("no score", self.no_score),
                ("no clock", self.no_clock),
            )
            if count
        )
        line = f"{self.scored} of {self.played} completed games have control ({share})"
        if dropped:
            line += f" -- dropped {dropped}"
        if self.disagreed:
            line += f" -- {self.disagreed} DISAGREE WITH THE STORED RESULT"
        return line


def _add(left: SweepStats, right: SweepStats) -> SweepStats:
    return SweepStats(*(a + b for a, b in zip(left, right)))


_NOTHING = SweepStats(
    scored=0,
    incomplete=0,
    unmatched=0,
    no_score=0,
    no_clock=0,
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
    its plays reach the score it finished at (`_covers`), and whether the two
    sources name the same winner (`_disagrees`).
    """
    return {
        game.game_id: game
        for season in seasons
        for week in season.weeks
        for game in week.games
        if game.completed
    }


def _final_from_plays(plays: Sequence[Play]) -> tuple[int, int] | None:
    """The score the plays end at, as (home, away).

    Taken as a maximum rather than off the last play: the scores are
    cumulative and non-decreasing, and the last row of a game is often an
    administrative one (END GAME) whose columns can be null. A game with no
    scores at all -- which the play table does contain -- has no final.
    """
    home = [play.home_score for play in plays if play.home_score is not None]
    away = [play.away_score for play in plays if play.away_score is not None]
    if not home or not away:
        return None
    return max(home), max(away)


def _disagrees(final: tuple[int, int], played: Game) -> bool:
    """Do the plays and the stored result disagree about who *won*?

    A weaker question than `_covers`, and a much more alarming answer. Play-by-
    play that stops early falls short of the final score without contradicting
    it; this is the case where the two sources name different winners, which
    they should not be able to do -- `lucky_ones` orients a curve by the play
    table's `home_score`/`away_score`, ESPN's own labelling, and `Game.home`
    comes from the same place.

    Compared by sign, and only when both sides are decisive, so a tie on
    either end isn't read as a contradiction.
    """
    from_plays = final[0] - final[1]
    from_result = played.home_score - played.away_score
    if not from_plays or not from_result:
        return False
    return (from_plays > 0) != (from_result > 0)


def _covers(final: tuple[int, int], played: Game) -> bool:
    """Do the plays run all the way to the game's final score?

    The coverage test, and it is an equality rather than a tolerance on
    purpose. A game the plays only half cover still produces a perfectly
    well-formed control number, and that number describes a different game --
    one where somebody led at halftime -- so the safe answer to "did the
    play-by-play cover this" is the strict one. A game that fails here keeps
    its real score, which is what every game without plays already does.

    If this turns out to reject games it shouldn't, the sweep's own report is
    where that shows up: `incomplete` counts them, and it is printed next to
    the games that came through.
    """
    return final == (played.home_score, played.away_score)


async def _sweep_week(
    league: str,
    source: PlaySource,
    year: int,
    week: int,
    played: Mapping[str, Game],
) -> tuple[dict[str, GameControl], SweepStats]:
    """One week's plays, reduced to one float per game.

    The plays go out of scope with this call, which is the point of doing it
    per week rather than assembling a season and scoring it.
    """
    plays = await source.load_week(league, year, week)
    if not plays:
        return {}, _NOTHING

    model = MODELS[league]
    control: dict[str, GameControl] = {}
    incomplete = unmatched = no_score = no_clock = disagreed = 0
    for game in group_by_game(plays):
        result = played.get(game.game_id)
        if result is None:
            unmatched += 1
            continue
        final = _final_from_plays(game.plays)
        if final is None:
            no_score += 1
            continue
        if _disagrees(final, result):
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
        if not _covers(final, result):
            incomplete += 1
            continue
        scored = model.game_control(game)
        if scored is None:
            no_clock += 1
            continue
        control[game.game_id] = GameControl(home=scored.home, seconds=scored.seconds)

    return control, _NOTHING._replace(
        scored=len(control),
        incomplete=incomplete,
        unmatched=unmatched,
        no_score=no_score,
        no_clock=no_clock,
        disagreed=disagreed,
    )


async def sweep(
    league: str, seasons: Sequence[Season], source: PlaySource
) -> tuple[dict[str, GameControl], SweepStats]:
    """Control for every game of `seasons` whose play-by-play covers it.

    Weeks run concurrently up to `_MAX_CONCURRENT_WEEKS`; a week nobody has
    processed comes back empty rather than raising, which is the normal state
    of the rest of a season in September.
    """
    played = completed_games(seasons)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_WEEKS)

    async def one(year: int, week: int):
        async with semaphore:
            return await _sweep_week(league, source, year, week, played)

    results = await asyncio.gather(
        *(one(year, week) for year, week in weeks_in(seasons))
    )

    control: dict[str, GameControl] = {}
    stats = _NOTHING._replace(played=len(played))
    for week_control, week_stats in results:
        control.update(week_control)
        stats = _add(stats, week_stats)
    return control, stats._replace(scored=len(control))


def write(league: str, fit: ControlFit, games: Mapping[str, GameControl]) -> None:
    """Save the index. Compact and key-sorted -- it's ~16,000 entries."""
    path = game_control_path(league)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = GameControlFile(league=league, fit=fit, games=dict(games))
    path.write_text(json.dumps(document.model_dump(mode="json"), sort_keys=True))


async def build(
    league: str,
    seasons: Sequence[Season],
    source: PlaySource,
    rebuild: bool = False,
) -> SweepStats | None:
    """Bring a league's index up to date, and say what happened.

    Three outcomes, decided by the stored `ControlFit`:

    - no index, or one built by a different fit: sweep every season. A
      different fit means every number in the file was produced by a model
      this build doesn't have, and topping that up would leave one index
      holding two models' opinions.
    - the same fit: sweep only the most recent season and merge. History is
      immutable while the model is, and this is the case a weekly run hits --
      about twenty weeks rather than four hundred.
    - the same fit, and `rebuild`: sweep everything anyway.

    Returns None when there was nothing to do, which can't currently happen
    (the newest season is always re-swept) but is what a caller should expect
    if a "nothing changed" case is ever added.
    """
    if not seasons:
        raise ValueError(f"No seasons for {league}; nothing to sweep")

    fit = current_fit(league)
    stored = read_game_control_file(league)
    stale = stored is None or stored.fit != fit
    full = rebuild or stale

    if full:
        reason = (
            "rebuild asked for"
            if stored is not None and not stale
            else "no index yet"
            if stored is None
            else f"built by a different fit ({stored.fit.run_id})"
        )
        print(f"{league}: full sweep -- {reason}")
        scope = list(seasons)
        existing: dict[str, GameControl] = {}
    else:
        latest = max(season.year for season in seasons)
        print(f"{league}: index is current for {fit.run_id}; refreshing {latest}")
        scope = [season for season in seasons if season.year == latest]
        existing = dict(stored.games) if stored else {}

    control, stats = await sweep(league, scope, source)
    # The refresh merges rather than replaces: `scope` is one season, and the
    # rest of the file is the same fit's answer for seasons that can't change.
    merged = {**existing, **control}
    write(league, fit, merged)
    print(f"  {stats}")
    print(f"  wrote {len(merged)} game(s) to {game_control_path(league)}")
    return stats
