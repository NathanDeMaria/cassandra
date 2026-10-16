"""A component replay against the published success model, game by game.

    python compare.py cm_NAME [against]

FBS v FBS games, 2014-2021 (the search's window) and 2022-2026 (held out),
paired per game with a t on the differences:

- brier on the win probability;
- the margin through a kernel fit on each model's own win probabilities
  through 2021 (`cassandra.prob_to_margin.kernel`, what a release carries),
  log loss and RPS -- the like-for-like comparison;
- the component model's own Normal, discretized to whole points, log loss --
  what reading the distribution straight off the parts is worth against
  the published model's kernel.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from cassandra.prob_to_margin.kernel import MarginKernelFitter

sys.path.insert(0, str(Path(__file__).parent.parent / "success_rate" / "combine"))
from distribution import per_game  # noqa: E402

OUT = Path.home() / ".cassandra/success_rate/units/ncaafb_out"
TRAIN_THROUGH = 2021


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / f"{name}.parquet").drop_duplicates("game_id")
    df["team1_mov"] = (df.home_score - df.away_score).astype(float)
    return df.set_index("game_id")


def kernel_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    train = df[df.year <= TRAIN_THROUGH]
    kernel = MarginKernelFitter().fit_df(train[["team1_win_prob", "team1_mov"]])
    scored = per_game(kernel, df.team1_win_prob.clip(1e-6, 1 - 1e-6).to_numpy(), df.team1_mov.to_numpy())
    scored["brier"] = ((df.team1_win_prob - (df.team1_mov > 0)) ** 2).to_numpy()
    return scored


def native_log_loss(df: pd.DataFrame) -> np.ndarray:
    m, mu, sd = df.team1_mov.to_numpy(), df.mu.to_numpy(), df.sd.to_numpy()
    p = norm.cdf((m + 0.5 - mu) / sd) - norm.cdf((m - 0.5 - mu) / sd)
    return -np.log(np.maximum(p, 1e-300))


def line(label: str, diff: np.ndarray, unit: float) -> str:
    t = diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))
    return f"{label} {diff.mean() * unit:+7.2f} (t {t:+5.1f})"


def main() -> None:
    name = sys.argv[1]
    against = sys.argv[2] if len(sys.argv) > 2 else "batch_nc_success_cap_p"
    a, b = load(name), load(against)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    sa, sb = kernel_scores(a), kernel_scores(b)
    has_mu = "mu" in a and a.mu.notna().any()
    native = native_log_loss(a) if has_mu else None
    print(f"{name} vs {against}: brier 1e-4, log loss 1e-3 nats, RPS 1e-2; negative favors {name}")
    for label, rows in (("2014-21", a.year.between(2014, TRAIN_THROUGH)),
                        ("held out", a.year > TRAIN_THROUGH),
                        ("2014-26", a.year >= 2014)):
        rows = (rows & a.fbs).to_numpy()
        parts = [f"{label:9s} n {rows.sum():5d}",
                 f"brier {sa['brier'][rows].mean():.6f} vs {sb['brier'][rows].mean():.6f}",
                 line("|", sa["brier"][rows] - sb["brier"][rows], 1e4),
                 line("| kernel ll", sa["log_loss"][rows] - sb["log_loss"][rows], 1e3),
                 line("| kernel rps", sa["rps"][rows] - sb["rps"][rows], 1e2)]
        if native is not None:
            parts.append(line("| own Normal ll vs their kernel", native[rows] - sb["log_loss"][rows], 1e3))
        print(" ".join(parts))
    if has_mu:
        rows = (a.fbs & (a.year >= 2014)).to_numpy()
        slope = np.polyfit(a.mu.to_numpy()[rows], a.team1_mov.to_numpy()[rows], 1)[0]
        z = (a.team1_mov - a.mu) / a.sd
        print(f"margin slope on mu {slope:.3f}; z sd {z[rows].std():.3f}; "
              f"within 1 sd {np.mean(np.abs(z[rows]) < 1):.3f}, within 2 {np.mean(np.abs(z[rows]) < 2):.3f}")
    per_year = pd.Series(sa["brier"] - sb["brier"], index=a.index)[(a.fbs & (a.year >= 2014)).to_numpy()]
    by = per_year.groupby(a.year[(a.fbs & (a.year >= 2014)).to_numpy()]).mean() * 1e4
    print("brier by season, 1e-4:", " ".join(f"{y}:{v:+.1f}" for y, v in by.items()))


if __name__ == "__main__":
    main()
