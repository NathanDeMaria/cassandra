"""Paired brier between replays, in 1e-4, on the same games.

    python compare.py LEAGUE BASE RUN [RUN ...]

NFL runs are `mixed/out/NAME.parquet` (every game); ncaafb runs are the
harness's `units/ncaafb_out/NAME.parquet`, scored as its search is, on FBS v
FBS games. Each RUN against BASE: through 2021 (the refits' window), held
out 2022 on, 2014 on, a side's games 1-4 and 5 on, with t on the per-game
differences and how many seasons of the window went RUN's way.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home() / ".cassandra/success_rate"
DIRS = {"nfl": HOME / "mixed/out", "ncaafb": HOME / "units/ncaafb_out"}


def load(league: str, name: str) -> pd.DataFrame:
    df = pd.read_parquet(DIRS[league] / f"{name}.parquet").reset_index(drop=True)
    # A side's game of the season, over every game it played, in date order.
    sides = pd.concat([
        pd.DataFrame({"row": df.index, "year": df.year, "team": df.home_team, "date": df.date}),
        pd.DataFrame({"row": df.index, "year": df.year, "team": df.away_team, "date": df.date}),
    ]).sort_values(["date", "row"], kind="stable")
    sides["n"] = sides.groupby(["year", "team"]).cumcount() + 1
    df["game_no"] = sides.groupby("row").n.min()
    if league == "ncaafb":
        df = df[df.fbs]
    df = df[df.year >= 2014].copy()
    df["loss"] = (df.team1_win_prob - df.team1_win) ** 2
    return df.set_index("game_id")


def line(label: str, d: pd.Series, years: pd.Series) -> str:
    if d.empty:
        return f"  {label:12s} -"
    t = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() else 0.0
    by = d.groupby(years).mean()
    return (f"  {label:12s} {d.mean() * 1e4:+6.2f}  t {t:+5.2f}  "
            f"{(by < 0).sum()}/{len(by)} seasons  n {len(d)}")


def main(league: str, base: str, runs: list[str]) -> None:
    b = load(league, base)
    print(f"{league}, against {base} (brier 2014 on {b.loss.mean():.6f})")
    for name in runs:
        r = load(league, name)
        both = b.index.intersection(r.index)
        d = r.loss.loc[both] - b.loss.loc[both]
        y = b.year.loc[both]
        g = b.game_no.loc[both]
        print(f"{name} (brier 2014 on {r.loss.loc[both].mean():.6f})")
        print(line("<=2021", d[y <= 2021], y[y <= 2021]))
        print(line("held out", d[y > 2021], y[y > 2021]))
        print(line("2014 on", d, y))
        print(line("games 1-4", d[g <= 4], y[g <= 4]))
        print(line("games 5+", d[g > 4], y[g > 4]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
