"""Does each estimate's uncertainty belong in the prediction?

    python uncertainty.py ncaafb|nfl

The published units model predicts from a logistic of the blended rating gap
at one fitted scale: no team's deviation reaches the probability. Under the
model's own Gaussian algebra the margin is

    N(mu, sigma^2 + c^2 (v_home + v_away))

with `v` a team's posterior variance, so a game between two teams the model
barely knows is called closer than one between two it knows well. Which `v`
is the question: the record's alone (what `_blended` hands on now), or the
record and units combined, `1 / (1/d^2 + w * earned)`.

From `stack.py`'s recorded ingredients (the published run), each variant's
scale -- one multiplier on the gap, one on the variance -- is fit on games
through 2021 and scored on 2022 on, against the published probabilities.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

from cassandra.predictor import PredictorConfig

LEAGUE = sys.argv[1]
OUT = Path.home() / ".cassandra/success_rate/combine"
FITTED = {
    "ncaafb": "glicko_margin_units_roster_result.json",
    "nfl": "glicko_margin_units_offseason_result.json",
}
TRAIN_THROUGH = 2021


def main() -> None:
    knobs = PredictorConfig.model_validate_json(
        (Path.home() / ".cassandra/models" / LEAGUE / FITTED[LEAGUE]).read_text()).params
    c, sigma, w = knobs["points_per_rating"], knobs["obs_sd"], knobs["unit_weight"]
    df = pd.read_parquet(OUT / f"stack_{LEAGUE}.parquet")
    df = df[df.fbs & (df.home_score != df.away_score)].copy()
    win = df.team1_win.astype(float).to_numpy()

    blended, parent_var, combined_var = {}, {}, {}
    for side in ("h", "a"):
        precision = 1 / df[f"d_{side}"] ** 2
        earned = w * df[f"info_{side}_published"]
        unit = df[f"anchor_{side}"] + df[f"u_{side}_published"]
        blended[side] = (precision * df[f"r_{side}"] + earned * unit) / (precision + earned)
        parent_var[side] = df[f"d_{side}"] ** 2
        combined_var[side] = 1 / (precision + earned)
    mu = (c * (blended["h"] + df.edge - blended["a"])).to_numpy()
    variances = {
        "none (fixed scale)": np.zeros(len(df)),
        "the record's": (c**2 * (parent_var["h"] + parent_var["a"])).to_numpy(),
        "record + units combined": (c**2 * (combined_var["h"] + combined_var["a"])).to_numpy(),
    }
    # Each team's game number in its season, for the early-season slice.
    order = df.sort_values("date")
    count = {}
    early = {}
    for g in order.itertuples():
        n = []
        for team in (g.home_team, g.away_team):
            count[g.year, team] = count.get((g.year, team), 0) + 1
            n.append(count[g.year, team])
        early[g.game_id] = min(n) <= 4
    df["early"] = df.game_id.map(early)

    train = (df.year <= TRAIN_THROUGH).to_numpy()
    test = ~train
    published = df.team1_win_prob_published.to_numpy()
    base = (published - win) ** 2
    print(f"{LEAGUE}: {train.sum()} games through {TRAIN_THROUGH}, {test.sum()} after;"
          f" sigma {sigma:.2f}, c {c:.4f}, unit_weight {w:.3f}")
    print(f"{'variance in the prediction':28s} {'held out':>9s} {'vs published':>13s} {'t':>6s}"
          f" {'games 1-4':>10s} {'later':>8s}   fitted")
    for name, var in variances.items():
        def probability(z, rows):
            gain, spread = np.exp(z)
            return norm.cdf(gain * mu[rows] / np.sqrt(sigma**2 + spread * var[rows]))

        def loss(z):
            return np.mean((probability(z, train) - win[train]) ** 2)

        z = optimize.minimize(loss, [0.0, 0.0], method="Nelder-Mead",
                              options={"xatol": 1e-4, "fatol": 1e-9}).x
        if not var.any():
            z[1] = 0.0
        p = probability(z, test)
        diff = (p - win[test]) ** 2 - base[test]
        t = diff.mean() / diff.std() * np.sqrt(len(diff))
        is_early = df.early.to_numpy()[test]
        print(f"{name:28s} {np.mean((p - win[test]) ** 2):9.6f} {diff.mean() * 1e4:+12.2f}e-4"
              f" {t:+6.2f} {diff[is_early].mean() * 1e4:+9.2f} {diff[~is_early].mean() * 1e4:+8.2f}"
              f"   gain {np.exp(z[0]):.3f} variance x{np.exp(z[1]):.2f}")
    sd_parent = np.sqrt(variances["the record's"][test])
    sd_combined = np.sqrt(variances["record + units combined"][test])
    print(f"\nrating uncertainty on the gap, points (held out): record's median {np.median(sd_parent):.2f}"
          f" (10-90% {np.percentile(sd_parent, 10):.2f}-{np.percentile(sd_parent, 90):.2f}),"
          f" combined median {np.median(sd_combined):.2f}; margin noise sigma {sigma:.2f}")


if __name__ == "__main__":
    main()
