"""The coach-change deviations, paired against the same knobs without them.

Slices: 2006+ (the facts' span), held-out 2022+, and a new-coach team's
games -- all of them, and its first four.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cassandra.offseason import OffseasonFacts

OUT = Path.home() / ".cassandra/success_rate/units/out"


def loss(df):
    return (df.team1_win_prob - df.team1_win) ** 2


def coach_games(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    facts = OffseasonFacts.for_league("nfl")
    changed = {(t, y) for y in range(2006, 2027) for t, f in facts.seasons(y) if f.coach_departure}
    d = df.sort_values("date")
    n = {}
    any_new = []
    early = []
    for row in d.itertuples():
        hits = []
        for team in (row.home_team, row.away_team):
            n[(team, row.year)] = n.get((team, row.year), 0) + 1
            if (team, row.year) in changed:
                hits.append(n[(team, row.year)])
        any_new.append(bool(hits))
        early.append(any(h <= 4 for h in hits))
    return (pd.Series(any_new, index=d.index).reindex(df.index),
            pd.Series(early, index=d.index).reindex(df.index))


def paired(a: str, b: str) -> None:
    da, db = pd.read_parquet(OUT / f"{a}.parquet"), pd.read_parquet(OUT / f"{b}.parquet")
    d = loss(da) - loss(db)
    new, early = coach_games(da)
    for label, mask in (("2006+", da.year >= 2006), ("2022+", da.year >= 2022),
                        ("new coach", (da.year >= 2006) & new), ("  games 1-4", (da.year >= 2006) & early)):
        x = d[mask]
        by = x.groupby(da.year[mask]).mean()
        print(f"  {a} - {b} {label:>11}: {1e4 * x.mean():+6.2f}e-4 over {mask.sum():4d}  "
              f"t {x.mean() / (x.std() / np.sqrt(len(x))):+.2f}  seasons better {int((by < 0).sum())}/{len(by)}")


if __name__ == "__main__":
    pairs = [arg.split(":") for arg in sys.argv[1:]] or [("coach_b75", "fit_b75"), ("coach_epa", "fit_epa_weekly")]
    for a, b in pairs:
        paired(a, b)
