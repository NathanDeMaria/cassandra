"""Building the EPA index: which number the sweep takes off a game.

The same shape as `cassandra.game_control_build`, around a different line.
`cassandra.pbp_sweep` walks the plays and decides which games can be trusted;
what's here is the call that turns one game into EPA per play, the file it
lands in, and the rule for when that file has to be rebuilt.

Two fits behind it rather than one, which is the only real difference from
the control build. `epa_per_play` reads the league's expected points model to
price each snap and its win probability model to say how much the game that
snap happened in was still in doubt, and the two retrain separately -- so
`EpaFit` carries both run ids and a sweep is stale if either moves.

The clip is passed explicitly rather than left to the package default. It is
one of the two things cassandra chooses about these numbers (the other is
which average to keep), it is recorded in the header, and a default that
moved underneath a stored index without moving the header is exactly the
failure the header exists to prevent. `lucky_ones.epa.DEFAULT_CLIP` is still
where the value comes from and where the measurement behind it is written
down -- this pins it, it doesn't second-guess it.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from endgame.types import Season
from lucky_ones import MODELS, GamePlays
from lucky_ones.epa import DEFAULT_CLIP, PlayEPA
from lucky_ones.plays import PlaySource
from lucky_ones.points import FIRST_LEGIBLE_SEASON

from cassandra.pbp_sweep import (
    SweepStats,
    fit_differences,
    lucky_ones_revision,
)
from cassandra.pbp_sweep import (
    sweep as sweep_plays,
)
from cassandra.predictor.epa import (
    EPA_READING,
    EpaFile,
    EpaFit,
    epa_path,
    read_epa_file,
)
from cassandra.predictor.types import GameEpa

# What one play is allowed to contribute, either way. `lucky_ones`' measured
# default -- swept against split-half prediction on held-out seasons, where it
# sits on a plateau from 2.5 to 4 -- named here because the header records it
# and because a sweep should say what it used rather than inherit it silently.
CLIP = DEFAULT_CLIP

# The exponent on the win-probability weighting behind the weighted pair.
# Passed explicitly for the reason the clip is -- the header records it, so
# a change upstream is found as a stale index rather than merged into one --
# and named here rather than taken from the package because the two have
# disagreed. `lucky_ones` shipped 2.0, measured on how well a whole-game
# number describes the game while it was in doubt; what cassandra does with
# the number is rate teams, and on that question every step past 0.5 costs
# sample for nothing (walk-forward R^2 on the next margin, ncaafb 2014-2025:
# 0.336 unweighted, 0.334 at 0.5, 0.321 at 1, 0.294 at 2). A decided snap
# carries about seven tenths of a live one's signal, and a quarter of college
# snaps are decided. 0.5 is where the weight a decided snap gets matches
# what it is worth. The package default moves to the same number in its
# next release; this is pinned so the header says what the index holds
# either way.
WEIGHT_POWER = 0.5

# The first season the index covers. The feed before it is legible now --
# see `lucky_ones.points.FIRST_LEGIBLE_SEASON` for what changed -- but the
# expected points fit starts here, and measured on what it recovers, the
# seasons before it read as a noisier copy of the margin that adds nothing
# to a rating built on the scoreboard (walk-forward, ncaafb 2006-2013: R^2
# 0.4086 with EPA against 0.4088 without, where 2014-2025 gains 0.004). A
# third of the index was those seasons, and a search over them lands at a
# whisper of the weight the later ones earn. Recorded in `EpaFit.since` so
# an index built to a different floor is found stale rather than merged.
SINCE = FIRST_LEGIBLE_SEASON

# What a success is, and the snaps a success rate is over. Named in the
# header because it is the third choice made here about these numbers, and a
# change to it -- counting penalties, say -- rewrites every success rate while
# the fits behind them stay identical.
#
# A success is a dropback or a run that added expected points: the sign of
# the snap's raw EPA, which the clip can't move. The denominator is the
# offense's dropbacks and runs -- sacks, scrambles, interceptions and fumbles
# included, since those are the plays a defense stopped -- and nothing else:
# punts and field goals are a decision about the drive rather than a down
# the defense won, a penalty that wiped the play out has no play in it, and
# a kneel or a spike is the clock. Measured on the NFL 2006-2026, those are
# a tenth of regulation snaps; what is left succeeds 45% of the time.
#
# Two rates off those snaps: every snap once, and each weighted by the
# competitiveness `WEIGHT_POWER` sets, as the two EPA readings are. ncaafb's
# units want the second -- a third of a mismatch is backups against backups
# -- and the NFL's the first; see `cassandra.predictor.unit_glicko`.
#
# And the flat rate split by the kind of down, dropbacks and runs apart, for
# a reader (`is_dropback` says which is which).
SUCCESS = (
    "epa>0, dropbacks and runs, no kneels or spikes; unweighted and weighted;"
    " dropbacks (passes, sacks, scrambles by their text) and runs apart"
)

# Kneels and spikes, off the play text: both are filed as an ordinary run or
# incompletion. Never a bare "spike" -- Takeo and Brandon Spikes made a lot
# of tackles between them.
_CLOCK_PLAY = re.compile(r"\bkneels?\b|\bkneel down\b|\btook a knee\b|\bspiked?\s+the\s+ball\b")

# The play types a dropback or a run is filed under, as substrings of the
# lowered type: "Pass Reception", "Rushing Touchdown", "Sack", "Pass
# Interception Return", "Fumble Recovery (Own)", "Safety".
_DOWN_TYPES = ("pass", "rush", "sack", "interception", "fumble", "safety")

# Special teams, which a punt or kick return touchdown and a blocked kick are
# filed as too.
_KICKS = ("punt", "field goal")

# A dropback by its type -- "Pass Reception", "Passing Touchdown", "Sack",
# "Interception Return Touchdown" -- or by its text: a scramble is filed as a
# run, and a fumble or a safety names the play it came off.
_DROPBACK_TYPES = re.compile(r"pass|sack|interception")
_SCRAMBLE = re.compile(r"\bscrambles?\b|\bscrambled\b")
_DROPBACK_TEXT = re.compile(r"\bpass\b|\bsacked\b|\bscrambles?\b|\bscrambled\b")


def current_fit(league: str) -> EpaFit:
    """What an index built right now would be built by.

    Raises `KeyError` for a league `lucky_ones` ships no fit for, and
    `FileNotFoundError` for one with a win probability fit and no expected
    points fit -- see `EPA_LEAGUES`.
    """
    model = MODELS[league]
    return EpaFit(
        lucky_ones=lucky_ones_revision(),
        run_id=model.run_id,
        ep_run_id=model.expected_points_release.run_id,
        clip=CLIP,
        reading=EPA_READING,
        weight_power=WEIGHT_POWER,
        since=SINCE,
        success=SUCCESS,
    )


def is_down(play_type: str | None, text: str | None) -> bool:
    """Whether a snap is a dropback or a run -- one a success rate is over.

    A null type is not one, unlike `lucky_ones`' snap filter: that filter
    keeps what it can't place because a snap left out of expected points is
    a hole in the curve, where one left out of a success rate is a slightly
    smaller denominator.
    """
    lowered = (play_type or "").lower()
    if not lowered or lowered == "penalty" or any(kick in lowered for kick in _KICKS):
        return False
    if _CLOCK_PLAY.search((text or "").lower()):
        return False
    return any(kind in lowered for kind in _DOWN_TYPES)


def is_dropback(play_type: str | None, text: str | None) -> bool:
    """Whether a down `is_down` keeps was a dropback rather than a run.

    By the play as it was filed, which is how it ended rather than what was
    called: a pass, a sack or an interception by its type, and a "Rush" the
    quarterback scrambled on by its text. The NFL's feed says "scrambles";
    the college feed never does, so there every scramble is a run.
    """
    lowered = (play_type or "").lower()
    words = (text or "").lower()
    if _DROPBACK_TYPES.search(lowered) or _SCRAMBLE.search(words):
        return True
    return ("fumble" in lowered or "safety" in lowered) and bool(_DROPBACK_TEXT.search(words))


class _Success(NamedTuple):
    home: float | None
    away: float | None
    home_downs: int
    away_downs: int
    home_weighted: float | None
    away_weighted: float | None
    home_dropbacks: tuple[float | None, int]
    away_dropbacks: tuple[float | None, int]
    home_runs: tuple[float | None, int]
    away_runs: tuple[float | None, int]


def _success(game: GamePlays, plays: Sequence[PlayEPA]) -> _Success:
    """Each offense's success rate, unweighted and weighted, and its dropbacks and runs.

    And the flat rate over each kind of down apart, with how many of each.
    """
    by_id = {play.play_id: play for play in game.plays}
    won = {True: 0, False: 0}
    downs = {True: 0, False: 0}
    won_weight = {True: 0.0, False: 0.0}
    weight = {True: 0.0, False: 0.0}
    # By side, then dropback or not.
    kind_won = {(side, dropback): 0 for side in (True, False) for dropback in (True, False)}
    kind_downs = dict(kind_won)
    for snap in plays:
        play = by_id.get(snap.play_id)
        if play is None or not is_down(play.play_type, play.text):
            continue
        side = snap.offense_is_home
        kind = (side, is_dropback(play.play_type, play.text))
        downs[side] += 1
        won[side] += snap.epa > 0
        kind_downs[kind] += 1
        kind_won[kind] += snap.epa > 0
        weight[side] += snap.weight
        won_weight[side] += snap.weight * (snap.epa > 0)

    def rate(side: bool, dropback: bool) -> tuple[float | None, int]:
        n = kind_downs[side, dropback]
        return (kind_won[side, dropback] / n if n else None), n

    return _Success(
        home=won[True] / downs[True] if downs[True] else None,
        away=won[False] / downs[False] if downs[False] else None,
        home_downs=downs[True],
        away_downs=downs[False],
        home_weighted=won_weight[True] / weight[True] if weight[True] else None,
        away_weighted=won_weight[False] / weight[False] if weight[False] else None,
        home_dropbacks=rate(True, True),
        away_dropbacks=rate(False, True),
        home_runs=rate(True, False),
        away_runs=rate(False, False),
    )


def _score(league: str, game: GamePlays) -> GameEpa | None:
    """One game's EPA per play, both readings, and its success rates.

    The unweighted reading -- `home_unweighted` and `away_unweighted`, every
    regulation snap counted once -- is what `home` and `away` hold, for the
    reason `GameEpa` gives: it is the one to add up across a season, and a
    rating model blending it into the scoreboard is a season being added up.
    The weighted pair off the same call goes in beside it, because a model
    rating an offense on how it played wants the game as it was contested;
    `weight_power` is passed explicitly so the header can say what it was.

    None only when the *flat* reading is missing, which is an offense with
    no snaps. The weighted reading can be missing on its own -- a side whose
    every snap came with the game decided -- and that is a game with a flat
    reading and no weighted one, which is what the tuple's None means.

    The success rates come off the same snaps' raw EPA, over the dropbacks
    and runs `is_down` picks out, once unweighted and once by the same
    weights as the weighted EPA, and unweighted over each kind apart
    (`is_dropback`); see `SUCCESS`.
    """
    scored = MODELS[league].epa_per_play(game, clip=CLIP, weight_power=WEIGHT_POWER)
    if scored.home_unweighted is None or scored.away_unweighted is None:
        return None
    success = _success(game, scored.plays)
    return GameEpa(
        home=scored.home_unweighted,
        away=scored.away_unweighted,
        home_plays=scored.home_plays,
        away_plays=scored.away_plays,
        home_weighted=scored.home,
        away_weighted=scored.away,
        home_weight=scored.home_weight,
        away_weight=scored.away_weight,
        home_success=success.home,
        away_success=success.away,
        home_success_plays=success.home_downs,
        away_success_plays=success.away_downs,
        home_success_weighted=success.home_weighted,
        away_success_weighted=success.away_weighted,
        home_dropback_success=success.home_dropbacks[0],
        away_dropback_success=success.away_dropbacks[0],
        home_dropback_plays=success.home_dropbacks[1],
        away_dropback_plays=success.away_dropbacks[1],
        home_run_success=success.home_runs[0],
        away_run_success=success.away_runs[0],
        home_run_plays=success.home_runs[1],
        away_run_plays=success.away_runs[1],
    )


async def sweep(
    league: str, seasons: Sequence[Season], source: PlaySource
) -> tuple[dict[str, GameEpa], SweepStats]:
    """EPA for every game of `seasons` whose play-by-play covers it."""
    return await sweep_plays(league, seasons, source, lambda game: _score(league, game))


def write(league: str, fit: EpaFit, games: Mapping[str, GameEpa]) -> None:
    """Save the index. Compact and key-sorted -- it's ~16,000 entries."""
    path = epa_path(league)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = EpaFile(league=league, fit=fit, games=dict(games))
    path.write_text(json.dumps(document.model_dump(mode="json"), sort_keys=True))


async def build(
    league: str,
    seasons: Sequence[Season],
    source: PlaySource,
    rebuild: bool = False,
) -> SweepStats | None:
    """Bring a league's index up to date, and say what happened.

    Three outcomes, decided by the stored `EpaFit`, exactly as the control
    build decides them:

    - no index, or one built by a different fit: sweep every season. A
      different fit means every number in the file was produced by models
      this build doesn't have, and topping that up would leave one index
      holding two models' opinions.
    - the same fit: sweep only the most recent season and merge. History is
      immutable while the models are, and this is the case a weekly run hits.
    - the same fit, and `rebuild`: sweep everything anyway.

    Returns None when there was nothing to do, which can't currently happen
    (the newest season is always re-swept) but is what a caller should expect
    if a "nothing changed" case is ever added.
    """
    if not seasons:
        raise ValueError(f"No seasons for {league}; nothing to sweep")
    # Applied before anything else so "the most recent season" and "every
    # season" both mean the seasons the index covers, and a league whose
    # cache reaches back further than the floor doesn't sweep the rest.
    seasons = [season for season in seasons if season.year >= SINCE]
    if not seasons:
        raise ValueError(f"No seasons for {league} since {SINCE}; nothing to sweep")

    fit = current_fit(league)
    stored = read_epa_file(league)
    stale = stored is None or stored.fit != fit
    full = rebuild or stale

    if full:
        reason = (
            "rebuild asked for"
            if stored is not None and not stale
            else "no index yet"
            if stored is None
            else f"built by a different fit ({fit_differences(stored.fit, fit)})"
        )
        print(f"{league}: full sweep -- {reason}")
        scope = list(seasons)
        existing: dict[str, GameEpa] = {}
    else:
        latest = max(season.year for season in seasons)
        print(f"{league}: index is current for {fit.run_id}; refreshing {latest}")
        scope = [season for season in seasons if season.year == latest]
        existing = dict(stored.games) if stored else {}

    epa, stats = await sweep(league, scope, source)
    # The refresh merges rather than replaces: `scope` is one season, and the
    # rest of the file is the same fits' answer for seasons that can't change.
    merged = {**existing, **epa}
    write(league, fit, merged)
    print(f"  {stats}")
    print(f"  wrote {len(merged)} game(s) to {epa_path(league)}")
    return stats
