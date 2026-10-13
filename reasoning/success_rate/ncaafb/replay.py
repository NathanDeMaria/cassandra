"""Replay ncaafb glicko_margin_units_roster with success rate in the units' reading.

The production predictor, given the swept index (`../units/sweep_index.py
ncaafb`, success rates included) instead of the shared file, which has none
yet and would read as EPA alone. One seasons pickle and one private copy of
the roster model's warm-up priors, so every variant differs only in what it
is given.

    python replay.py prep                        seasons from s3, priors from the replay cache
    python replay.py run NAME [knob=value ...]   one replay, to ncaafb_out/NAME.parquet

Scored as the search scores it, brier on FBS v FBS games, split into the
refit's window (through 2021) and the held-out seasons.
"""

import asyncio
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import asdict
from functools import cache
from pathlib import Path

import pandas as pd

from cassandra.objective import _in_fbs
from cassandra.odds import OddsDatabase
from cassandra.predictor import PredictorConfig
from cassandra.predictor.epa import EpaIndex
from cassandra.predictor.opponent_prior import OpponentPriorManager
from cassandra.predictor.unit_glicko import SUCCESS_SCALE, UnitMarginGlickoPredictor
from cassandra.save_predictions import join_with_odds, prepared_for_replay

DATA = Path.home() / ".cassandra/success_rate/units"
SEASONS = DATA / "ncaafb_seasons.pkl"
PRIORS = DATA / "ncaafb_units_priors.json"
CACHED_PRIORS = Path.home() / ".cassandra/replays/ncaafb/glicko_margin_units_roster.priors.json"
FITTED = Path.home() / ".cassandra/models/ncaafb/glicko_margin_units_roster_result.json"
OUT = DATA / "ncaafb_out"
#: Which swept index to read: `sweep_index.py`'s, or `sweep_weighted.py`'s.
INDEX = DATA / os.environ.get("NCAAFB_INDEX", "ncaafb_index.pkl")
TRAIN_THROUGH = 2021


def prep() -> None:
    from endgame_aws import Config

    from cassandra.save_predictions import read_rated_seasons

    seasons = asyncio.run(read_rated_seasons("ncaafb", Config.init_from_file().bucket))
    SEASONS.write_bytes(pickle.dumps(seasons))
    shutil.copy(CACHED_PRIORS, PRIORS)
    print(len(seasons), "seasons", seasons[0].year, "-", seasons[-1].year)


@cache
def _inputs():
    named, namer = prepared_for_replay("ncaafb", pickle.loads(SEASONS.read_bytes()))
    games = pickle.loads(INDEX.read_bytes())
    if os.environ.get("NCAAFB_EPA") == "weighted":
        # The units on the competitiveness-weighted EPA pair instead of the
        # flat one, to tell weighting apart from success rate.
        games = {
            g: e._replace(home=e.home_weighted, away=e.away_weighted)
            for g, e in games.items()
            if e.home_weighted is not None and e.away_weighted is not None
        }
    if "NCAAFB_SUCCESS_SCALE" in os.environ:
        # A reading whose own EPA-scale constants differ from the flat rate's
        # in `SUCCESS_SCALE`: rewritten so that the code's constants land it
        # where its own would, `scale * s + level` either way.
        scale, level = map(float, os.environ["NCAAFB_SUCCESS_SCALE"].split(","))
        flat_scale, flat_level = SUCCESS_SCALE["ncaafb"]["unweighted"]

        def rescaled(rate: float | None) -> float | None:
            return None if rate is None else (scale * rate + level - flat_level) / flat_scale

        games = {
            g: e._replace(home_success=rescaled(e.home_success), away_success=rescaled(e.away_success))
            for g, e in games.items()
        }
    index = EpaIndex(games)
    return named, namer, index


BASE = PredictorConfig.model_validate_json(FITTED.read_text()).params


def _replay(predictor: UnitMarginGlickoPredictor, seasons: list) -> pd.DataFrame:
    _, namer, _ = _inputs()
    df = pd.DataFrame([asdict(p) for p in join_with_odds(
        predictor, seasons, OddsDatabase({}), post_callbacks=False, namer=namer)])
    df["fbs"] = [
        _in_fbs(h, int(y)) and _in_fbs(a, int(y))
        for h, a, y in zip(df.home_team, df.away_team, df.year)
    ]
    return df


def run(knobs: dict) -> pd.DataFrame:
    """The whole history, 2002 on, as the evaluate stage replays it."""
    named, _, index = _inputs()
    predictor = UnitMarginGlickoPredictor(
        "ncaafb",
        **{**BASE, **knobs},
        opponent_prior_manager=OpponentPriorManager("ncaafb", path=PRIORS),
        game_epa=index,
    )
    return _replay(predictor, named)


#: The first season the index has a reading for. Before it the units have
#: nothing to step on and no sides exist, so no reading knob can touch the
#: state a replay reaches by then.
FIRST_READING = 2014


@cache
def _warm() -> tuple[UnitMarginGlickoPredictor, pd.DataFrame]:
    """The published model through 2013, rolled over into 2014."""
    named, _, index = _inputs()
    predictor = UnitMarginGlickoPredictor(
        "ncaafb",
        **BASE,
        opponent_prior_manager=OpponentPriorManager("ncaafb", path=PRIORS),
        game_epa=index,
    )
    early = _replay(predictor, [s for s in named if s.year < FIRST_READING])
    assert not predictor.state_dict()["sides"], "sides before the first reading"
    return predictor, early


def run_from_2014(
    knobs: dict,
    through: int = 2026,
    predictor_class: type[UnitMarginGlickoPredictor] = UnitMarginGlickoPredictor,
) -> pd.DataFrame:
    """The same replay resumed from the 2013 snapshot, and stopped after `through`.

    The search's objective needs nothing after `TRAIN_THROUGH`, and nothing
    before 2014 differs between knobs; the two cuts leave about a third of
    the games.
    """
    named, _, index = _inputs()
    warm, early = _warm()
    predictor = predictor_class.from_state_dict({
        **warm.state_dict(),
        **knobs,
        "opponent_prior_manager": OpponentPriorManager("ncaafb", path=PRIORS),
        "game_epa": index,
    })
    # What a rollover leaves that a state dict doesn't carry: the clock, and
    # the teams whose quarterback shift waits on their first game.
    predictor._season = warm._season
    predictor._qb_pending = set(warm._qb_pending)
    late = _replay(predictor, [s for s in named if FIRST_READING <= s.year <= through])
    return pd.concat([early, late], ignore_index=True)


def scores(df: pd.DataFrame) -> dict[str, float]:
    loss = (df.team1_win_prob - df.team1_win) ** 2
    fbs = df.fbs.to_numpy()
    return {
        "train": float(loss[fbs & (df.year <= TRAIN_THROUGH)].mean()),
        "held_out": float(loss[fbs & (df.year > TRAIN_THROUGH)].mean()),
        "since_2014": float(loss[fbs & (df.year >= 2014)].mean()),
        "all_fbs": float(loss[fbs].mean()),
    }


def save(name: str, df: pd.DataFrame, knobs: dict, **extra) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{name}.parquet")
    (OUT / f"{name}.json").write_text(json.dumps({"knobs": knobs, **scores(df), **extra}, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    else:
        name = sys.argv[2]
        knobs = {k: float(v) for k, v in (arg.split("=") for arg in sys.argv[3:])}
        start = time.time()
        df = run(knobs)
        save(name, df, knobs)
        print(name, knobs, {k: round(v, 6) for k, v in scores(df).items()},
              f"({time.time() - start:.0f}s)", flush=True)

