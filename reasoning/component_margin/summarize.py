"""One line per saved replay: FBS brier by window, against the published success model.

    python summarize.py PREFIX [PREFIX ...]
"""

import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path.home() / ".cassandra/success_rate/units/ncaafb_out"
PUBLISHED = "batch_nc_success_cap_p"


def windows(name: str) -> dict[str, float]:
    df = pd.read_parquet(OUT / f"{name}.parquet")
    loss = (df.team1_win_prob - df.team1_win) ** 2
    out = {}
    for label, lo, hi in (("14-21", 2014, 2021), ("22-26", 2022, 2026), ("14-26", 2014, 2026)):
        rows = df.fbs & df.year.between(lo, hi)
        out[label] = float(loss[rows].mean())
    return out


def main() -> None:
    base = windows(PUBLISHED)
    print(f"{'run':24s} " + "  ".join(f"{w:>15s}" for w in base) + "   (1e-4 vs published)   margin ll held")
    for prefix in sys.argv[1:]:
        for path in sorted(OUT.glob(f"cm_{prefix}*.parquet")):
            name = path.stem
            w = windows(name)
            meta = json.loads((OUT / f"{name}.json").read_text())
            cells = "  ".join(f"{w[k]:.6f} {1e4 * (w[k] - base[k]):+5.1f}" for k in base)
            print(f"{name:24s} {cells}   {meta.get('margin_ll_held_out', float('nan')):.4f}")


if __name__ == "__main__":
    main()
