"""After a lopsided game, does the favorite beat the model in its next competitive game?

If lopsided games mark favorites down for winning by less than a linear
expectation, the next game against a real opponent (|expected| < 14) finds
them better than rated: a positive mean surprise, growing with how lopsided
the last game was and how many such games came before.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mismatch_efficiency import BINS, OUT, clustered, team_games  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "batch_nc_success_cap_p"
df = pd.read_parquet(OUT / f"{name}.parquet").drop_duplicates("game_id")
tg = team_games(df)
tg["bin"] = pd.cut(tg.expected, [-200, -14, 14, 21, 28, 35, 45, 200], right=False)
close_next = tg.next_expected.abs() < 14
for label, rows in (("2002-21", tg.year <= 2021), ("2022-26", tg.year >= 2022),
                    ("FBS teams, 2014-26", (tg.year >= 2014) & tg.fbs)):
    sub = tg[rows & close_next]
    print(f" {label}: next game competitive (|expected| < 14), its mean surprise by this game's expected margin")
    for b_, s in sub.groupby("bin", observed=True):
        se = s.next_surprise.groupby(s.team + s.year.astype(str)).mean().std() / np.sqrt(s.groupby(["team", "year"]).ngroups)
        print(f"   this game {str(b_):12s} n {len(s):6d}  next surprise {s.next_surprise.mean():+5.2f} ± {s.next_surprise.std() / np.sqrt(len(s)):.2f}")
# Cumulative: how many lopsided wins (expected >= 28) a team has had this season before a competitive game.
tg = tg.sort_values(["team", "year", "date"])
tg["lopsided"] = (tg.expected >= 28).astype(int)
tg["lopsided_before"] = tg.groupby(["team", "year"]).lopsided.cumsum()  # including game t
comp = tg[close_next & (tg.year <= 2021)]
print(" 2002-21, next competitive game's surprise by lopsided favorite games so far this season")
for k, s in comp.groupby(comp.lopsided_before.clip(upper=4)):
    print(f"   {k}{'+' if k == 4 else ' '} lopsided so far  n {len(s):6d}  next surprise {s.next_surprise.mean():+5.2f} ± {s.next_surprise.std() / np.sqrt(len(s)):.2f}")
comp = tg[close_next & (tg.year >= 2022)]
print(" 2022-26, the same")
for k, s in comp.groupby(comp.lopsided_before.clip(upper=4)):
    print(f"   {k}{'+' if k == 4 else ' '} lopsided so far  n {len(s):6d}  next surprise {s.next_surprise.mean():+5.2f} ± {s.next_surprise.std() / np.sqrt(len(s)):.2f}")
