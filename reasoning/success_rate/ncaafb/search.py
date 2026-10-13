"""Refit the units' reading knobs for one success weight, by FBS brier through 2021.

    python search.py NAME WEIGHT [FROM]     WEIGHT a number, or "free" to search it too

Every weight gets the same knobs and budget, EPA (weight 0) included, so a
weight that wins has won on the same terms: the eight in the reading's units,
and `unit_weight` -- how far the margin trusts the units at all, which on
ncaafb is 0.59 against the NFL's 1, and which a better reading should earn
more of. Starts from the published roster fit, or FROM's knobs.
Nelder-Mead in logs (shifted, so a knob at 0 can leave it), the weights on
their own scale.

Writes ncaafb_out/NAME.parquet (the whole history) and NAME.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).parent))
from replay import BASE, OUT, TRAIN_THROUGH, run_from_2014, save, scores  # noqa: E402

#: knob: (shift, the step a knob at 0 takes in the first simplex)
POSITIVE = {
    "points_per_epa": (1.0, 20.0),
    "play_sd": (0.05, 0.5),
    "offense_initial_sd": (0.01, 0.05),
    "defense_initial_sd": (0.01, 0.05),
    "offense_weekly_sd_increase": (0.01, 0.03),
    "defense_weekly_sd_increase": (0.01, 0.03),
    "offense_season_sd_increase": (0.02, 0.1),
    "defense_season_sd_increase": (0.02, 0.1),
}
SHARES = ["unit_weight"]


def main(name: str, weight: str, start_from: str | None) -> None:
    start = dict(BASE)
    if start_from:
        start.update(json.loads((OUT / f"{start_from}.json").read_text())["knobs"])
    names = [*POSITIVE, *SHARES]
    if weight == "free":
        names.append("success_weight")
        start.setdefault("success_weight", 0.75)
        if not start["success_weight"]:
            start["success_weight"] = 0.75
    else:
        start["success_weight"] = float(weight)

    def encode(knobs: dict) -> np.ndarray:
        return np.array([
            np.log(knobs[k] + POSITIVE[k][0]) if k in POSITIVE else knobs[k] for k in names
        ])

    def decode(z: np.ndarray) -> dict:
        out = {k: start[k] for k in [*names, "success_weight"]}
        for k, v in zip(names, z):
            out[k] = (max(0.0, float(np.exp(v) - POSITIVE[k][0])) if k in POSITIVE
                      else float(np.clip(v, 0.0, 1.0)))
        return out

    x0 = {k: start[k] for k in names}
    vertices = [encode(x0)]
    for k in names:
        moved = dict(x0)
        if k in POSITIVE:
            moved[k] = x0[k] * 1.4 if x0[k] > 0 else POSITIVE[k][1]
        else:
            moved[k] = x0[k] + 0.15 if x0[k] <= 0.85 else x0[k] - 0.15
        vertices.append(encode(moved))

    seen = []
    began = time.time()

    def objective(z: np.ndarray) -> float:
        loss = scores(run_from_2014(decode(z), through=TRAIN_THROUGH))["train"]
        seen.append(loss)
        if len(seen) % 20 == 0:
            print(f"{name} {len(seen)} {loss:.6f} best {min(seen):.6f} "
                  f"({(time.time() - began) / len(seen):.0f}s/probe)", flush=True)
        return loss

    fit = optimize.minimize(
        objective, vertices[0], method="Nelder-Mead",
        options={"maxfev": 50 * len(names), "initial_simplex": np.array(vertices),
                 "fatol": 2e-7, "xatol": 1e-3},
    )
    knobs = decode(fit.x)
    df = run_from_2014(knobs)
    save(name, df, knobs, evals=len(seen), weight=weight)
    print(name, "done", {k: round(v, 6) for k, v in scores(df).items()},
          {k: round(v, 4) for k, v in knobs.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
