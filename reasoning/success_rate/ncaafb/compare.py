"""Two replays, paired game by game, on the slices the spread of opponents lives in.

    python compare.py A:B [A:B ...]

Brier of A minus B, in 1e-4, on games from 2014 (the reading's span), with
a t from the per-game differences and the count of seasons A won. Slices:

- FBS v FBS (the objective), and the held-out seasons of it;
- by how lopsided the game was, from the published model's own pregame
  win probability, so every pair is cut the same way;
- power v power, power v Group of Five, Group of Five v Group of Five;
- FBS v anything below, which the objective never sees but which is most of
  what an FBS team's units read in September;
- each FBS team's games 1-4 and the rest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cassandra.residuals import team_tiers

OUT = Path.home() / ".cassandra/success_rate/units/ncaafb_out"
POWER = {"Southeastern", "Big Ten", "Big 12", "Atlantic Coast", "Pac-12"}
FAVORITE_BINS = [0.5, 0.65, 0.8, 0.9, 0.97, 1.0]


def _labels(df: pd.DataFrame) -> pd.DataFrame:
    tier = team_tiers(df, "ncaafb")
    assert tier is not None

    def group(team: str, year: int) -> str:
        found = tier(team, int(year))
        if found is None or found[0] != "FBS":
            return "below"
        return "power" if found[1] in POWER or team.startswith("Notre Dame") else "G5"

    home = [group(t, y) for t, y in zip(df.home_team, df.year)]
    away = [group(t, y) for t, y in zip(df.away_team, df.year)]
    pairs = ["v".join(sorted(p)) for p in zip(home, away)]
    published = pd.read_parquet(OUT / "published.parquet").set_index("game_id").team1_win_prob
    favorite = df.game_id.map(published)
    favorite = np.maximum(favorite, 1 - favorite)
    order = df.sort_values("date").index
    count: dict[tuple[str, int], int] = {}
    early = pd.Series(False, index=df.index)
    for i in order:
        row = df.loc[i]
        flags = []
        for team in (row.home_team, row.away_team):
            count[(team, row.year)] = count.get((team, row.year), 0) + 1
            flags.append(count[(team, row.year)] <= 4)
        early[i] = any(flags)
    return pd.DataFrame({"pair": pairs, "favorite": favorite, "early": early}, index=df.index)


def paired(a: str, b: str, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    da = pd.read_parquet(OUT / f"{a}.parquet").set_index("game_id")
    db = pd.read_parquet(OUT / f"{b}.parquet").set_index("game_id")
    db = db.loc[da.index]
    da = da[da.year >= 2014]
    db = db.loc[da.index]
    if labels is None:
        labels = _labels(da.reset_index()).set_index(da.index)
    d = ((da.team1_win_prob - da.team1_win) ** 2) - ((db.team1_win_prob - db.team1_win) ** 2)
    fbs = da.fbs
    rows = [
        ("FBS v FBS", fbs),
        ("  held out", fbs & (da.year >= 2022)),
        ("  games 1-4", fbs & labels.early),
        ("  later", fbs & ~labels.early),
        *[(f"  favorite {lo:.2f}-{hi:.2f}", fbs & (labels.favorite > lo) & (labels.favorite <= hi))
          for lo, hi in zip(FAVORITE_BINS, FAVORITE_BINS[1:])],
        ("  power v power", labels.pair == "powervpower"),
        ("  G5 v power", labels.pair == "G5vpower"),
        ("  G5 v G5", labels.pair == "G5vG5"),
        ("FBS v below", labels.pair.isin(["belowvpower", "G5vbelow"])),
    ]
    print(f"\n{a} - {b}")
    for label, mask in rows:
        x = d[mask]
        by = x.groupby(da.year[mask]).mean()
        t = x.mean() / (x.std() / np.sqrt(len(x)))
        print(f"  {label:>22}: {1e4 * x.mean():+6.2f}e-4 over {len(x):5d}  t {t:+5.2f}  "
              f"seasons better {int((by < 0).sum())}/{len(by)}")
    return labels


if __name__ == "__main__":
    labels = None
    for arg in sys.argv[1:]:
        a, b = arg.split(":")
        labels = paired(a, b, labels)
