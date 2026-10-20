"""Do the conclusions hold when a model is graded on its margin distribution?

    python distribution.py ncaafb|nfl

Everything else in `combine/` grades a win probability by brier. A release
also carries a margin kernel (`cassandra.prob_to_margin.kernel`): a
cumulative logit on every integer margin with the win probability's logit
as its one covariate, which is what a spread is read against. Two
questions:

1. **The comparisons, on margins.** For each pair the write-up compared on
   brier -- the success-rate models against the published ones, the
   discounted margin against its control -- each model's kernel is fit on
   its games through 2021 and scores 2022 on: the margin log loss (minus
   the log probability of the margin the game finished at) and the ranked
   probability score, beside brier, paired game by game.

2. **The ratings' uncertainty, on margins.** The kernel reads the win
   probability alone, so two games called at 70% get the same margin
   distribution whether the model knows both teams or neither. Here the
   kernel's logits are scaled by `exp(-gamma * z)`, `z` the game's rating
   uncertainty on the gap (log sd about its median): gamma > 0 widens the
   distribution where the ratings are unsure. Fit on top of the kernel
   through 2021, scored after, with the record's deviation alone and with
   the record and units combined (`stack.py`'s recording).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from cassandra.predictor import PredictorConfig
from cassandra.prob_to_margin.kernel import MarginKernelFitter, _sigmoid

LEAGUE = sys.argv[1]
DATA = Path.home() / ".cassandra/success_rate/units"
COMBINE = Path.home() / ".cassandra/success_rate/combine"
OUT = DATA / ("ncaafb_out" if LEAGUE == "ncaafb" else "out")
TRAIN_THROUGH = 2021
FIRST = 2014 if LEAGUE == "ncaafb" else 2006
FITTED = {
    "ncaafb": "glicko_margin_units_roster_result.json",
    "nfl": "glicko_margin_units_offseason_result.json",
}
#: (name, against), each a replay saved by the harness.
PAIRS = {
    "ncaafb": [
        ("nc_p_off", "published_p", "success model vs published roster"),
        ("disc_s", "disc_s_obs", "discounted margin on the success model vs its control"),
        ("disc_nc_ps", "disc_nc_obs_ps", "discounted margin on the roster model vs its control"),
    ],
    "nfl": [
        ("batch_nfl_success_p", "base_p", "success model vs published offseason"),
        ("disc_nfl", "disc_nfl_obs", "discounted margin vs its control"),
    ],
}


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"{name}.parquet").drop_duplicates("game_id")
    if "fbs" not in df:
        df["fbs"] = True
    df["team1_mov"] = (df.home_score - df.away_score).astype(float)
    return df.set_index("game_id")


def per_game(kernel, probs: np.ndarray, margins: np.ndarray, scale=None) -> dict[str, np.ndarray]:
    """Each game's margin log loss and RPS under `kernel`, its logits scaled if asked."""
    k = kernel.max_margin
    eta = kernel.slope * np.log(probs / (1 - probs))
    logits = kernel.cutpoints[None, :] - eta[:, None]
    if scale is not None:
        logits = logits * scale[:, None]
    cdfs = np.concatenate([_sigmoid(logits), np.ones((len(probs), 1))], axis=1)
    pmfs = np.diff(cdfs, axis=1, prepend=0.0)
    cells = np.clip(np.rint(margins).astype(int), -k, k) + k
    realized = np.arange(2 * k)[None, :] >= cells[:, None]
    return {
        "log_loss": -np.log(np.maximum(pmfs[np.arange(len(cells)), cells], 1e-300)),
        "rps": ((cdfs[:, :-1] - realized) ** 2).sum(axis=1),
    }


def kernel_on(df: pd.DataFrame):
    train = df[(df.year <= TRAIN_THROUGH)]
    return MarginKernelFitter().fit_df(train[["team1_win_prob", "team1_mov"]])


def compare() -> None:
    print(f"\n1. comparisons on the margin, each model's kernel fit through {TRAIN_THROUGH}")
    print(f"{'pair':58s} {'slice':9s} {'log loss':>14s} {'RPS':>14s} {'brier':>13s}")
    for name, against, label in PAIRS[LEAGUE]:
        a, b = load(name), load(against)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        scores = {}
        for tag, df in (("a", a), ("b", b)):
            kernel = kernel_on(df)
            scored = per_game(kernel, df.team1_win_prob.clip(1e-6, 1 - 1e-6).to_numpy(),
                              df.team1_mov.to_numpy())
            scored["brier"] = ((df.team1_win_prob - (df.team1_mov > 0)) ** 2).to_numpy()
            scores[tag] = scored
        for slice_name, rows in (
            ("held out", (a.year > TRAIN_THROUGH) & a.fbs),
            (f"{FIRST}-26", (a.year >= FIRST) & a.fbs),
        ):
            rows = rows.to_numpy()
            cells = []
            for metric, unit in (("log_loss", 1e3), ("rps", 1e2), ("brier", 1e4)):
                diff = scores["a"][metric][rows] - scores["b"][metric][rows]
                t = diff.mean() / diff.std() * np.sqrt(len(diff))
                cells.append(f"{diff.mean() * unit:+6.2f} (t {t:+5.1f})")
            print(f"{label:58s} {slice_name:9s} " + " ".join(cells))
    print("  (log loss in 1e-3 nats, RPS in 1e-2, brier in 1e-4; negative is better)")


def uncertainty() -> None:
    knobs = PredictorConfig.model_validate_json(
        (Path.home() / ".cassandra/models" / LEAGUE / FITTED[LEAGUE]).read_text()).params
    c, w = knobs["points_per_rating"], knobs["unit_weight"]
    df = pd.read_parquet(COMBINE / f"stack_{LEAGUE}.parquet")
    df = df[df.fbs].copy()
    df["team1_mov"] = (df.home_score - df.away_score).astype(float)
    parent, combined = 0.0, 0.0
    for side in ("h", "a"):
        precision = 1 / df[f"d_{side}"] ** 2
        earned = w * df[f"info_{side}_published"]
        parent = parent + df[f"d_{side}"] ** 2
        combined = combined + 1 / (precision + earned)
    probs = df.team1_win_prob_published.clip(1e-6, 1 - 1e-6).to_numpy()
    margins = df.team1_mov.to_numpy()
    train = (df.year <= TRAIN_THROUGH).to_numpy()
    test = ~train
    kernel = MarginKernelFitter().fit_df(
        df.loc[train, ["team1_win_prob_published", "team1_mov"]].rename(
            columns={"team1_win_prob_published": "team1_win_prob"}))
    base = per_game(kernel, probs[test], margins[test])
    print(f"\n2. the ratings' uncertainty in the margin kernel ({LEAGUE}, FBS v FBS for ncaafb)")
    print(f"   rating sd on the gap, held out: record's median {c * np.sqrt(np.median(parent[test])):.1f}"
          f" points (10-90% {c * np.sqrt(np.percentile(parent[test], 10)):.1f}-"
          f"{c * np.sqrt(np.percentile(parent[test], 90)):.1f})")
    print(f"{'uncertainty':26s} {'gamma':>7s} {'log loss':>14s} {'RPS':>14s}   held out, vs the kernel alone")
    for name, var in (("the record's", parent), ("record + units combined", combined)):
        z = 0.5 * np.log(var.to_numpy())
        z = z - np.median(z[train])

        def loss(gamma, rows):
            return per_game(kernel, probs[rows], margins[rows], np.exp(-gamma * z[rows]))["log_loss"].mean()

        gamma = optimize.minimize_scalar(lambda g: loss(g, train), bounds=(-3, 3), method="bounded").x
        scored = per_game(kernel, probs[test], margins[test], np.exp(-gamma * z[test]))
        cells = []
        for metric, unit in (("log_loss", 1e3), ("rps", 1e2)):
            diff = scored[metric] - base[metric]
            t = diff.mean() / diff.std() * np.sqrt(len(diff))
            cells.append(f"{diff.mean() * unit:+6.2f} (t {t:+5.1f})")
        print(f"{name:26s} {gamma:7.3f} " + " ".join(cells))


if __name__ == "__main__":
    compare()
    uncertainty()
