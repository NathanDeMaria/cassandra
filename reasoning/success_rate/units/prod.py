"""The same replays through the production knobs: success_weight and the coach-change deviations.

    python prod.py check                 production code reproduces the harness
    python prod.py coach NAME FROM       fit the three coach knobs on FROM's knobs
    python prod.py full NAME FROM        refit the eight reading knobs, the weight
                                         and the coach knobs together, from FROM

Reads the swept index from nfl_index.pkl (sweep_index.py), never the shared file.
"""

import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, str(Path(__file__).parent))
from replay import FITTED, PRIORS, ensure_priors, seasons  # noqa: E402

from cassandra.odds import OddsDatabase  # noqa: E402
from cassandra.predictor import PredictorConfig  # noqa: E402
from cassandra.predictor.epa import EpaIndex  # noqa: E402
from cassandra.predictor.opponent_prior import OpponentPriorManager  # noqa: E402
from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor  # noqa: E402
from cassandra.save_predictions import join_with_odds  # noqa: E402

DATA = Path.home() / ".cassandra/success_rate/units"
OUT = DATA / "out"
TRAIN_THROUGH = 2021
READING = ["points_per_epa", "play_sd", "offense_initial_sd", "defense_initial_sd",
           "offense_weekly_sd_increase", "defense_weekly_sd_increase",
           "offense_season_sd_increase", "defense_season_sd_increase"]
COACH = ["coach_change_rd", "coach_change_offense_sd", "coach_change_defense_sd"]
FLOOR = 1e-3

NAMED, NAMER = seasons()
ensure_priors(NAMED)
BASE = PredictorConfig.model_validate_json(FITTED.read_text()).params
INDEX = EpaIndex(pickle.loads((DATA / "nfl_index.pkl").read_bytes()))


def run(knobs: dict) -> pd.DataFrame:
    predictor = UnitMarginGlickoPredictor(
        "nfl", **{**BASE, **knobs},
        opponent_prior_manager=OpponentPriorManager("nfl", path=PRIORS),
        game_epa=INDEX,
    )
    return pd.DataFrame([asdict(p) for p in join_with_odds(
        predictor, NAMED, OddsDatabase({}), post_callbacks=False, namer=NAMER)])


def brier(df: pd.DataFrame, years=None) -> float:
    loss = (df.team1_win_prob - df.team1_win) ** 2
    return float(loss[df.year <= TRAIN_THROUGH].mean() if years is None else loss[years(df.year)].mean())


def from_run(name: str) -> dict:
    meta = json.loads((OUT / f"{name}.json").read_text())
    return {**meta["knobs"], "success_weight": meta.get("success_weight", meta.get("beta", 0.0))}


def search(name: str, start_knobs: dict, names: list[str], scale: dict[str, float]) -> None:
    def encode(k):
        return np.log(start_knobs.get(k, 0.0) + FLOOR) if k != "success_weight" else start_knobs[k]

    def decode(z):
        out = dict(start_knobs)
        for k, v in zip(names, z):
            out[k] = float(np.clip(v, 0, 1)) if k == "success_weight" else max(0.0, float(np.exp(v) - FLOOR))
        return out

    seen = []

    def objective(z):
        loss = brier(run(decode(z)))
        seen.append(loss)
        if len(seen) % 25 == 0:
            print(name, len(seen), f"{loss:.6f}", flush=True)
        return loss

    start = np.array([encode(k) for k in names])
    simplex = np.vstack([start, start + np.diag([scale.get(k, 0.35) for k in names])])
    fit = optimize.minimize(objective, start, method="Nelder-Mead",
                            options={"maxfev": 60 * len(names), "initial_simplex": simplex, "fatol": 2e-7})
    knobs = decode(fit.x)
    df = run(knobs)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / f"{name}.parquet")
    (OUT / f"{name}.json").write_text(json.dumps(
        {"beta": knobs.get("success_weight", 0.0), "since": 2014, "knobs": knobs, "train": fit.fun,
         "evals": len(seen)}, indent=2))
    print(name, "done", f"train {fit.fun:.6f}", {k: round(v, 4) for k, v in knobs.items()}, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        for label, knobs in (("published", {}), ("fit_b75", from_run("fit_b75")),
                             ("fit_epa_weekly", from_run("fit_epa_weekly"))):
            df = run(knobs)
            ref = OUT / f"{'base' if label == 'published' else label}.parquet"
            same = np.allclose(df.team1_win_prob, pd.read_parquet(ref).team1_win_prob, atol=1e-12)
            print(f"{label}: brier {((df.team1_win_prob - df.team1_win) ** 2).mean():.6f}, "
                  f"identical to the harness: {same}", flush=True)
    elif mode == "coach":
        name, start = sys.argv[2], from_run(sys.argv[3])
        start = {**start, "coach_change_rd": 30.0, "coach_change_offense_sd": 0.05,
                 "coach_change_defense_sd": 0.02}
        search(name, start, COACH, {"coach_change_rd": 1.0, "coach_change_offense_sd": 1.0,
                                    "coach_change_defense_sd": 1.0})
    elif mode == "full":
        name, start = sys.argv[2], from_run(sys.argv[3])
        search(name, start, READING + ["success_weight"] + COACH,
               {"success_weight": 0.15, **{k: 0.7 for k in COACH}})
