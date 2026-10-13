"""Fit the collision's six knobs by walk-forward predictive likelihood.

Every game is scored before it updates anything, so the objective is how
well the model would have predicted each game's successes from what it knew
the day before. 2006 is burn-in; the fit is on 2007-2025, and 2026 so far is
reported alongside as an out-of-sample read.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).parent))
from collide import Params, replay
from success import load_plays, team_games

OUT = Path.home() / ".cassandra" / "success_rate"
FIT = range(2007, 2026)


def load_games():
    games = team_games(load_plays())
    counts = games.groupby(["season", "offense"]).size()
    franchises = set(counts[counts >= 8].index.get_level_values("offense"))
    return games[games.offense.isin(franchises) & games.defense.isin(franchises)]


def score(history, seasons=FIT) -> float:
    return float(history.loc[history.season.isin(seasons), "loglik"].sum())


def main(mode: str) -> None:
    games = load_games()
    if mode == "time":
        for size in (24, 32, 48, 64):
            start = time.time()
            h = replay(games, Params(), size=size)
            print(f"grid {size}: {time.time() - start:.1f}s loglik {score(h):.3f}")
        return

    size = 24
    evals = []

    # `fit` is the season-constant model, searched from the defaults with
    # drift held at 0; `drift` starts from that fit and searches the weekly
    # drifts with everything else.
    if mode == "drift":
        seed = json.loads((OUT / "collide_params.json").read_text())
        start = Params(**{**seed, "drift_off": 0.005, "drift_def": 0.005}).to_vector()
        steps = np.diag([0.3, 0.3, 0.4, 0.4, 0.3, 0.02, 1.0, 1.0])
        name = "collide_params_drift.json"
    else:
        start = Params().to_vector()[:6]
        steps = np.diag([0.4, 0.4, 0.5, 0.5, 0.4, 0.03])
        name = "collide_params.json"
    pad = Params().to_vector()[6:]

    def full(v):
        return Params.from_vector(v if len(v) == 8 else np.r_[v, pad])

    def objective(v):
        params = full(v)
        ll = score(replay(games, params, size=size))
        evals.append(ll)
        if len(evals) % 10 == 0:
            print(len(evals), round(ll, 2), params, flush=True)
        return -ll

    simplex = np.vstack([start, start + steps])
    fit = optimize.minimize(objective, start, method="Nelder-Mead",
                            options={"xatol": 1e-3, "fatol": 0.05, "maxfev": 800,
                                     "initial_simplex": simplex})
    params = full(fit.x)
    print("fit", params, -fit.fun, fit.message, flush=True)
    (OUT / name).write_text(json.dumps({k: float(v) for k, v in params.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fit")
