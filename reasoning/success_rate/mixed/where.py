"""Where the mixed reading's ncaafb gain sits: after a blowout or a game against a lower tier?

    python where.py BASE RUN

FBS v FBS games from 2014, each tagged by the two sides' previous game this
season -- a blowout (margin 28+) or not, against a lower tier or not -- and
by a side's game of the season. Brier change, RUN less BASE, in 1e-4.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cassandra.objective import _in_fbs

DIR = Path.home() / ".cassandra/success_rate/units/ncaafb_out"


def tagged(name: str) -> pd.DataFrame:
    df = pd.read_parquet(DIR / f"{name}.parquet").reset_index(drop=True)
    df["home_fbs"] = [_in_fbs(t, int(y)) for t, y in zip(df.home_team, df.year)]
    df["away_fbs"] = [_in_fbs(t, int(y)) for t, y in zip(df.away_team, df.year)]
    sides = pd.concat([
        pd.DataFrame({"row": df.index, "year": df.year, "team": df.home_team, "date": df.date,
                      "margin": df.home_score - df.away_score, "opp_lower": df.home_fbs & ~df.away_fbs}),
        pd.DataFrame({"row": df.index, "year": df.year, "team": df.away_team, "date": df.date,
                      "margin": df.away_score - df.home_score, "opp_lower": df.away_fbs & ~df.home_fbs}),
    ]).sort_values(["date", "row"], kind="stable")
    by = sides.groupby(["year", "team"])
    sides["n"] = by.cumcount() + 1
    sides["prev_blowout"] = by.margin.shift(1).abs() >= 28
    sides["prev_lower"] = by.opp_lower.shift(1).astype("boolean").fillna(False).astype(bool)
    agg = sides.groupby("row").agg(n=("n", "min"), prev_blowout=("prev_blowout", "any"),
                                   prev_lower=("prev_lower", "any"))
    df = df.join(agg)
    df = df[df.fbs & (df.year >= 2014)].copy()
    df["loss"] = (df.team1_win_prob - df.team1_win) ** 2
    return df.set_index("game_id")


def main(base: str, run: str) -> None:
    b, r = tagged(base), tagged(run)
    d = (r.loss - b.loss).loc[b.index]
    print(f"{run} less {base}, FBS v FBS 2014 on, 1e-4")
    for label, mask in {
        "all": pd.Series(True, index=b.index),
        "game 1 (no previous game)": b.n == 1,
        "games 2-4": b.n.between(2, 4),
        "games 5+": b.n >= 5,
        "a side's previous game a blowout": b.prev_blowout,
        "  neither side (games 2+)": ~b.prev_blowout & (b.n > 1),
        "a side's previous game vs a lower tier": b.prev_lower,
        "  neither side (games 2+) ": ~b.prev_lower & (b.n > 1),
    }.items():
        x = d[mask]
        t = x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 and x.std() else 0.0
        print(f"  {label:42s} {x.mean() * 1e4:+6.2f}  t {t:+5.2f}  n {len(x)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
