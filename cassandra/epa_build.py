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
from collections.abc import Mapping, Sequence

from endgame.types import Season
from lucky_ones import MODELS, GamePlays
from lucky_ones.epa import DEFAULT_CLIP, DEFAULT_WEIGHT_POWER
from lucky_ones.plays import PlaySource

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
# The package default, and passed explicitly for the reason the clip is:
# the header records it, so a change to the default upstream is found as a
# stale index rather than merged into one.
WEIGHT_POWER = DEFAULT_WEIGHT_POWER


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
    )


def _score(league: str, game: GamePlays) -> GameEpa | None:
    """One game's EPA per play, both readings, or None if an offense had no snaps.

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
    """
    scored = MODELS[league].epa_per_play(game, clip=CLIP, weight_power=WEIGHT_POWER)
    if scored.home_unweighted is None or scored.away_unweighted is None:
        return None
    return GameEpa(
        home=scored.home_unweighted,
        away=scored.away_unweighted,
        home_plays=scored.home_plays,
        away_plays=scored.away_plays,
        home_weighted=scored.home,
        away_weighted=scored.away,
        home_weight=scored.home_weight,
        away_weight=scored.away_weight,
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
