"""Building the game control index: which number the sweep takes off a game.

Split from `cassandra.predictor.game_control` on purpose. That module is on
the replay path and reads a JSON file; the sweep behind this one imports
`lucky_ones.arrow` and pyarrow and moves ~190MB of parquet, and lives in the
`fit` group with the rest of the fitting stack. The artifact is the seam
between them.

Split again from `cassandra.pbp_sweep`, which is everything about walking the
plays that isn't about which number you keep -- the week loop, the coverage
tests, the counts a run prints. What's left here is one line of scoring
(`luck_adjusted_game_control`), the file it lands in, and the rule for when
that file has to be rebuilt. `cassandra.epa_build` is the same shape around a
different line.

Idempotency is keyed on the fit, not on the file existing. A stage that finds
its own `ControlFit` already stored refreshes only the season still being
played -- history can't change while the model doesn't -- and one that finds a
different fit rebuilds the league from scratch, because merging numbers from
two different win probability models into one index would produce a rating
nobody could reproduce.
"""

import json
from collections.abc import Mapping, Sequence

from endgame.types import Season
from lucky_ones import MODELS, GamePlays
from lucky_ones.plays import PlaySource

from cassandra.pbp_sweep import (
    SweepStats,
    fit_differences,
    lucky_ones_revision,
)
from cassandra.pbp_sweep import (
    sweep as sweep_plays,
)
from cassandra.pbp_sweep import (
    weeks_in as weeks_in,  # re-exported: the sweep's week order, for tests
)
from cassandra.predictor.game_control import (
    CONTROL_READING,
    ControlFit,
    GameControlFile,
    game_control_path,
    read_game_control_file,
)
from cassandra.predictor.types import GameControl


def current_fit(league: str) -> ControlFit:
    """What an index built right now would be built by.

    Raises `KeyError` for a league `lucky_ones` ships no fit for, which is
    every league that isn't football -- see `CONTROL_LEAGUES`.
    """
    return ControlFit(
        lucky_ones=lucky_ones_revision(),
        run_id=MODELS[league].run_id,
        reading=CONTROL_READING,
    )


def _score(league: str, game: GamePlays) -> GameControl | None:
    """One game's control, or None if there's no clock to average over.

    The luck-adjusted reading, not the realized one: the curve redrawn with
    the fumbles and the tipped balls split evenly rather than credited to
    whoever they fell to. It is None on exactly the games `game_control` is
    None on, so `SweepStats.unscored` still counts what it says.
    """
    scored = MODELS[league].luck_adjusted_game_control(game)
    if scored is None:
        return None
    return GameControl(home=scored.home, seconds=scored.seconds)


async def sweep(
    league: str, seasons: Sequence[Season], source: PlaySource
) -> tuple[dict[str, GameControl], SweepStats]:
    """Control for every game of `seasons` whose play-by-play covers it."""
    return await sweep_plays(
        league, seasons, source, lambda game: _score(league, game)
    )


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
            else f"built by a different fit ({fit_differences(stored.fit, fit)})"
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
