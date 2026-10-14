"""Refit the reading's own knobs for one reading, by brier on games through 2021.

    python search.py NAME BETA SINCE

Every reading gets the same eight knobs and the same budget, EPA included, so
a reading that wins has won on the same terms. The knobs are the ones in the
reading's units -- the scale, the noise, the deviations and both clocks'
increases; everything else stays at the published fit. Weekly increases are
searched from 0 up, which is the question of whether a unit should forget
during a season (an injury showing up in its results faster).
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, str(Path(__file__).parent))
from readings import build  # noqa: E402
from replay import FITTED, PRIORS, ensure_priors, seasons  # noqa: E402

from cassandra.odds import OddsDatabase  # noqa: E402
from cassandra.predictor import PredictorConfig  # noqa: E402
from cassandra.predictor.opponent_prior import OpponentPriorManager  # noqa: E402
from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor  # noqa: E402
from cassandra.save_predictions import join_with_odds  # noqa: E402

KNOBS = ["points_per_epa", "play_sd", "offense_initial_sd", "defense_initial_sd",
         "offense_weekly_sd_increase", "defense_weekly_sd_increase",
         "offense_season_sd_increase", "defense_season_sd_increase"]
FLOOR = 1e-3  # weekly/season increases are searched as log(v + FLOOR), so 0 is reachable
TRAIN_THROUGH = 2021
OUT = Path.home() / ".cassandra/success_rate/units/out"


def main(name: str, beta: float, since: int, pinned: dict | None = None) -> None:
    pinned = pinned or {}
    knobs_searched = [k for k in KNOBS if k not in pinned]
    named, namer = seasons()
    ensure_priors(named)
    base = PredictorConfig.model_validate_json(FITTED.read_text()).params
    index = build(beta, since)

    def run(knobs: dict) -> pd.DataFrame:
        predictor = UnitMarginGlickoPredictor(
            "nfl", **{**base, **knobs},
            opponent_prior_manager=OpponentPriorManager("nfl", path=PRIORS),
            game_epa=index,
        )
        return pd.DataFrame([asdict(p) for p in join_with_odds(
            predictor, named, OddsDatabase({}), post_callbacks=False, namer=namer)])

    def decode(z):
        return {k: max(0.0, float(np.exp(v) - (FLOOR if "increase" in k else 0)))
                for k, v in zip(knobs_searched, z)} | pinned

    start = np.array([np.log(base[k] + (FLOOR if "increase" in k else 0)) for k in knobs_searched])
    seen = []

    def objective(z):
        knobs = decode(z)
        df = run(knobs)
        loss = ((df.team1_win_prob - df.team1_win) ** 2)[df.year <= TRAIN_THROUGH].mean()
        seen.append(loss)
        if len(seen) % 25 == 0:
            print(name, len(seen), f"{loss:.6f}", flush=True)
        return loss

    simplex = np.vstack([start, start + np.diag([0.35] * len(knobs_searched))])
    fit = optimize.minimize(objective, start, method="Nelder-Mead",
                            options={"maxfev": 320, "initial_simplex": simplex, "fatol": 2e-7})
    knobs = decode(fit.x)
    df = run(knobs)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / f"{name}.parquet")
    (OUT / f"{name}.json").write_text(json.dumps({"beta": beta, "since": since, "knobs": knobs,
                                                  "train": fit.fun, "evals": len(seen)}, indent=2))
    print(name, "done", f"train {fit.fun:.6f}", {k: round(v, 4) for k, v in knobs.items()}, flush=True)


if __name__ == "__main__":
    pins = {k: float(v) for k, v in (arg.split("=") for arg in sys.argv[4:])}
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), pins)
