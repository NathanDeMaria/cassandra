"""Replay the NFL success model with its units on the pooled or the mixed success rate.

The published `glicko_margin_units_success` knobs (run 20261014-010003, fit
through 2021), on the units harness's seasons, priors and pinned data, given
either the harness's index (pooled) or `nfl_index.py`'s (mixed at weight W).
A mixed index is rescaled on the way in so the code's pooled constants land
it where its own would -- `scale * rate + level` either way.

    python nfl_replay.py run NAME INDEX [knob=value ...]
    python nfl_replay.py search NAME INDEX [--from RUN] [--knobs K,K,...]

INDEX is `pooled` or a mix weight (`0.65`). Writes out/NAME.parquet and
NAME.json beside the indexes, scored on every game through 2021 (the fit's
window) and after.
"""

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "units"))
from replay import FITTED, PRIORS, ensure_priors, seasons  # noqa: E402  (pins the data)

from cassandra.odds import OddsDatabase  # noqa: E402
from cassandra.predictor import PredictorConfig  # noqa: E402
from cassandra.predictor.epa import EpaIndex  # noqa: E402
from cassandra.predictor.opponent_prior import OpponentPriorManager  # noqa: E402
from cassandra.predictor.unit_glicko import (  # noqa: E402
    SUCCESS_SCALE,
    UnitMarginGlickoPredictor,
)
from cassandra.save_predictions import join_with_odds  # noqa: E402

HOME = Path.home() / ".cassandra/success_rate"
OUT = HOME / "mixed/out"
TRAIN_THROUGH = 2021
PUBLISHED = json.loads((HOME / "units/out/batch_nfl_success.json").read_text())["knobs"]
BASE = {**PredictorConfig.model_validate_json(FITTED.read_text()).params, **PUBLISHED}

#: knob: (shift for the log, first step from 0); the weights search on their own scale.
POSITIVE = {"points_per_epa": (1.0, 20.0), "play_sd": (0.05, 0.5), "qb_out_penalty": (1.0, 10.0),
            "offense_weekly_sd_increase": (0.005, 0.01), "offense_initial_sd": (0.01, 0.03)}
SHARES = {"success_weight", "defense_success_weight"}

NAMED, NAMER = seasons()
ensure_priors(NAMED)


@cache
def index(name: str) -> EpaIndex:
    if name == "pooled":
        return EpaIndex(pickle.loads((HOME / "units/nfl_index.pkl").read_bytes()))
    games = pickle.loads((HOME / f"mixed/nfl_index_w{name}.pkl").read_bytes())
    own = json.loads((HOME / f"mixed/nfl_index_w{name}.json").read_text())
    flat_scale, flat_level = SUCCESS_SCALE["nfl"]["unweighted"]

    def rescaled(rate):
        return None if rate is None else (own["scale"] * rate + own["level"] - flat_level) / flat_scale

    return EpaIndex({
        g: e._replace(home_success=rescaled(e.home_success), away_success=rescaled(e.away_success))
        for g, e in games.items()
    })


def run(index_name: str, knobs: dict) -> pd.DataFrame:
    predictor = UnitMarginGlickoPredictor(
        "nfl", **{**BASE, **knobs},
        opponent_prior_manager=OpponentPriorManager("nfl", path=PRIORS),
        game_epa=index(index_name),
    )
    return pd.DataFrame([asdict(p) for p in join_with_odds(
        predictor, NAMED, OddsDatabase({}), post_callbacks=False, namer=NAMER)])


def scores(df: pd.DataFrame) -> dict[str, float]:
    loss = (df.team1_win_prob - df.team1_win) ** 2
    return {
        "train": float(loss[df.year <= TRAIN_THROUGH].mean()),
        "held_out": float(loss[df.year > TRAIN_THROUGH].mean()),
        "since_2014": float(loss[df.year >= 2014].mean()),
    }


def save(name: str, df: pd.DataFrame, **meta) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{name}.parquet")
    (OUT / f"{name}.json").write_text(json.dumps({**meta, **scores(df)}, indent=2))
    print(name, {k: round(v, 6) for k, v in scores(df).items()}, flush=True)


def search(name: str, index_name: str, start: dict, names: list[str]) -> None:
    def encode(k, v):
        return np.log(v + POSITIVE[k][0]) if k in POSITIVE else v

    def decode(z):
        out = dict(start)
        for k, v in zip(names, z):
            out[k] = max(0.0, float(np.exp(v) - POSITIVE[k][0])) if k in POSITIVE else float(np.clip(v, 0, 1))
        return out

    x0 = np.array([encode(k, start[k]) for k in names])
    simplex = [x0]
    for i, k in enumerate(names):
        moved = dict(start)
        if k in POSITIVE:
            moved[k] = start[k] * 1.4 if start[k] > 0 else POSITIVE[k][1]
        else:
            moved[k] = start[k] + 0.15 if start[k] <= 0.85 else start[k] - 0.15
        simplex.append(np.array([encode(j, moved[j]) for j in names]))
    seen = []
    began = time.time()

    def objective(z):
        loss = scores(run(index_name, decode(z)))["train"]
        seen.append(loss)
        if len(seen) % 20 == 0:
            print(f"{name} {len(seen)} {loss:.6f} best {min(seen):.6f} "
                  f"({(time.time() - began) / len(seen):.1f}s/probe)", flush=True)
        return loss

    fit = optimize.minimize(objective, x0, method="Nelder-Mead",
                            options={"maxfev": 50 * len(names), "initial_simplex": np.array(simplex),
                                     "fatol": 2e-7, "xatol": 1e-3})
    knobs = decode(fit.x)
    moved = {k: knobs[k] for k in names}
    save(name, run(index_name, knobs), index=index_name, knobs=moved, evals=len(seen))
    print(name, "knobs", {k: round(v, 4) for k, v in moved.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "search"])
    parser.add_argument("name")
    parser.add_argument("index")
    parser.add_argument("--from", dest="start_from")
    parser.add_argument("--knobs", default="success_weight,points_per_epa,play_sd")
    parser.add_argument("set", nargs="*", default=[])
    args = parser.parse_args()
    start = dict(BASE)
    if args.start_from:
        start.update(json.loads((OUT / f"{args.start_from}.json").read_text())["knobs"])
    knobs = {k: float(v) for k, v in (pair.split("=") for pair in args.set)}
    start.update(knobs)
    if args.mode == "run":
        save(args.name, run(args.index, start), index=args.index, knobs=knobs)
    else:
        search(args.name, args.index, start, args.knobs.split(","))
