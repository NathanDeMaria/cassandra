"""What a summer's change does to each split: how much is kept, and how much is new.

    python summer_change.py nfl|ncaafb

Reads `split_persistence.py`'s tables. A correlation within a subgroup is
the wrong measure here -- fired coaches and new quarterbacks cluster on bad
teams, and a narrow range lowers a correlation whatever the persistence --
so each group gets the filter's own two numbers, net of measurement noise:

    next = keep * last + shift + new,   new ~ (0, R)

- `keep`: the slope of next season's true level on last season's, within
  the group (the offseason regression is 1 - keep);
- `new`: R over the side's true variance, the share of a spread that
  arrives over the summer -- what a season sd increase is for;
- `shift`: the group's mean move, in true spreads.

The noise of a season's estimate comes from the split halves (a half's
noise over two), the true variance from their covariance. Errors are a
bootstrap over seasons. Then, for each kind, new less no-new for both
numbers, and that difference for dropbacks less runs: whether a new
quarterback (or coach) unsettles one kind more than the other.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home() / ".cassandra/success_rate/mixed"
BOOT = 300


def _centered(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["a"] -= frame.groupby("year").a.transform("mean")
    frame["b"] -= frame.groupby("year").b.transform("mean")
    return frame


def _noise(halves: pd.DataFrame) -> tuple[float, float]:
    """(true variance, a season estimate's noise variance)."""
    h = _centered(halves)
    true = float(np.mean(h.a * h.b))
    half_noise = float((np.mean(h.a**2) + np.mean(h.b**2)) / 2) - true
    return true, half_noise / 2


def _group(pairs: pd.DataFrame, true: float, noise: float) -> tuple[float, float, float]:
    """keep, new variance over true variance, and shift in true spreads, for one group."""
    a, b = pairs.a, pairs.b
    cov = float(np.mean((a - a.mean()) * (b - b.mean())))
    var_a, var_b = float(a.var(ddof=0)), float(b.var(ddof=0))
    keep = cov / (var_a - noise)
    new = (var_b - noise - cov**2 / (var_a - noise)) / true
    shift = (b.mean() - keep * a.mean()) / np.sqrt(true)
    return keep, new, shift


def _stats(halves: pd.DataFrame, pairs: pd.DataFrame, flag: str) -> dict:
    out = {}
    pairs = _centered(pairs)
    for (part, side), p in pairs.groupby(["part", "side"]):
        true, noise = _noise(halves[(halves.part == part) & (halves.side == side)])
        for value in (True, False):
            g = p[p[flag] == value]
            out[part, side, value] = (*_group(g, true, noise), len(g))
    return out


def main(league: str) -> None:
    halves = pd.read_parquet(HOME / f"persistence_{league}_halves.parquet")
    pairs = pd.read_parquet(HOME / f"persistence_{league}_pairs.parquet")
    years = sorted(pairs.year.unique())
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 250)
    for flag, label in (("new_qb", "new week-one quarterback"), ("coach", "head coach left")):
        known = pairs[pairs[flag].notna()].copy()
        known[flag] = known[flag].astype(bool)
        point = _stats(halves, known, flag)
        boots = []
        for _ in range(BOOT):
            pick = rng.choice(years, len(years), replace=True)
            h = pd.concat([halves[halves.year == y].assign(year=i) for i, y in enumerate(pick)])
            p = pd.concat([known[known.year == y].assign(year=i) for i, y in enumerate(pick)])
            boots.append(_stats(h, p, flag))
        rows = []
        for (part, side, value), (keep, new, shift, n) in point.items():
            if not value:
                continue
            k0, n0, s0, m0 = point[part, side, False]
            dk = [b[part, side, True][0] - b[part, side, False][0] for b in boots]
            dn = [b[part, side, True][1] - b[part, side, False][1] for b in boots]
            rows.append({
                "part": part, "side": side, "n": f"{n}/{m0}",
                "keep": keep, "keep, none": k0, "Δkeep": keep - k0, "se": np.std(dk),
                "new": new, "new, none": n0, "Δnew": new - n0, "se ": np.std(dn),
                "shift": shift,
            })
        table = pd.DataFrame(rows).set_index(["part", "side"])
        print(f"\n{league}: {label} (n = with/without; keep and new net of noise; shift in true spreads)")
        print(table.round(3).to_string())
        print("  dropbacks less runs, offense:")
        for value in ("sr", "epa"):
            for i, name in ((0, "Δkeep"), (1, "Δnew")):
                diffs = [
                    (b[f"{value} dropback", "offense", True][i] - b[f"{value} dropback", "offense", False][i])
                    - (b[f"{value} run", "offense", True][i] - b[f"{value} run", "offense", False][i])
                    for b in boots
                ]
                est = ((point[f"{value} dropback", "offense", True][i] - point[f"{value} dropback", "offense", False][i])
                       - (point[f"{value} run", "offense", True][i] - point[f"{value} run", "offense", False][i]))
                print(f"    {value} {name}: {est:+.3f} (se {np.std(diffs):.3f})")


if __name__ == "__main__":
    main(sys.argv[1])
