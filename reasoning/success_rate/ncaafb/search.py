"""Refit some of the units' knobs, by FBS brier through 2021.

    python search.py NAME [--from RUN] [--knobs K,K,...] [--set K=V ...]

`--knobs` is what moves -- by default the eight in the reading's units and
`unit_weight`, how far the margin trusts the units at all -- and `--set`
pins the rest on top of RUN's knobs (the published roster fit by default),
`success_weight=free` meaning "search it from 0.75". Every reading gets the
same knobs and budget, EPA included, so one that wins has won on the same
terms; the narrow searches (`--knobs success_weight,points_per_epa,play_sd`
on the EPA refit's clocks) are the ones that don't overfit eight seasons.
Nelder-Mead in logs (shifted, so a knob at 0 can leave it), the weights on
their own scale.

Writes ncaafb_out/NAME.parquet (the whole history) and NAME.json.
"""

import argparse
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
    "anchor_epa_scale": (0.05, 0.2),
}
#: Knobs in [0, 1], searched on their own scale.
SHARES = {"unit_weight", "success_weight", "defense_success_weight"}
DEFAULT_KNOBS = [*list(POSITIVE)[:8], "unit_weight"]


def _value(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--from", dest="start_from")
    parser.add_argument("--knobs", default=",".join(DEFAULT_KNOBS))
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    start = dict(BASE)
    if args.start_from:
        start.update(json.loads((OUT / f"{args.start_from}.json").read_text())["knobs"])
    names = args.knobs.split(",")
    for pair in args.set:
        key, value = pair.split("=")
        if value == "free":
            names.append(key)
            start[key] = start.get(key) or 0.75
        else:
            start[key] = _value(value)
    for key in names:
        start.setdefault(key, 1.0 if key == "anchor_epa_scale" else 0.0)

    def encode(knobs: dict) -> np.ndarray:
        return np.array([
            np.log(knobs[k] + POSITIVE[k][0]) if k in POSITIVE else knobs[k] for k in names
        ])

    def decode(z: np.ndarray) -> dict:
        out = dict(start)
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
            print(f"{args.name} {len(seen)} {loss:.6f} best {min(seen):.6f} "
                  f"({(time.time() - began) / len(seen):.0f}s/probe)", flush=True)
        return loss

    fit = optimize.minimize(
        objective, vertices[0], method="Nelder-Mead",
        options={"maxfev": 50 * len(names), "initial_simplex": np.array(vertices),
                 "fatol": 2e-7, "xatol": 1e-3},
    )
    knobs = decode(fit.x)
    df = run_from_2014(knobs)
    save(args.name, df, knobs, evals=len(seen), searched=names)
    print(args.name, "done", {k: round(v, 6) for k, v in scores(df).items()},
          {k: (round(knobs[k], 4) if isinstance(knobs[k], float) else knobs[k]) for k in names},
          flush=True)


if __name__ == "__main__":
    main()
