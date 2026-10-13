"""Is a refit's error about schedule? Cross-tier bias and conference moves.

1. Group of Five v power games: the power side's mean outcome less its
   predicted win probability, per refit. A reading that lets a weak
   schedule flatter a team overrates the Group of Five here.
2. Brier against the EPA refit by season, and split by whether either team
   is in its first three seasons after changing conference -- what the
   2023-24 realignment would show up in.

    python bias.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compare import OUT, POWER, _labels  # noqa: E402

from cassandra.residuals import team_tiers  # noqa: E402

names = ["published", "nc_fit_epa", "nc_fit_wepa", "nc_fit_free", "nc_fit_ww_free", "nc_fit_wepa_ww_free"]
dfs = {n: pd.read_parquet(OUT / f"{n}.parquet").set_index("game_id") for n in names}
base = dfs["published"]
base = base[base.year >= 2014]
labels = _labels(base.reset_index()).set_index(base.index)
tier = team_tiers(base.reset_index(), "ncaafb")

def conf(team, year):
    t = tier(team, int(year))
    return None if t is None else t[1]

# Which teams changed conference from one season to the next (FBS only).
moved = set()
for (team, year) in {(t, y) for t, y in zip(base.home_team, base.year)} | {(t, y) for t, y in zip(base.away_team, base.year)}:
    c, p = conf(team, year), conf(team, year - 1)
    if c and p and c != p:
        for y in range(year, year + 3):
            moved.add((team, y))   # first three seasons in the new conference
base_mov = np.array([(h, y) in moved or (a, y) in moved for h, a, y in zip(base.home_team, base.away_team, base.year)])

print("1. cross-tier bias: G5 v power games, mean(outcome - predicted) for the power side, 1e-2 win prob")
mask = labels.pair == "G5vpower"
for n, d in dfs.items():
    d = d.loc[base.index]
    g = d[mask]
    # power side's perspective
    tiers_h = [tier(t, int(y)) for t, y in zip(g.home_team, g.year)]
    hp = np.array([t is not None and (t[1] in POWER or h.startswith("Notre Dame")) for t, h in zip(tiers_h, g.home_team)])
    p_power = np.where(hp, g.team1_win_prob, 1 - g.team1_win_prob)
    w_power = np.where(hp, g.team1_win, ~g.team1_win).astype(float)
    r = w_power - p_power
    by = pd.Series(r).groupby(g.year.to_numpy()).mean()
    print(f"  {n:>22}: {100*r.mean():+5.2f} (se {100*r.std()/np.sqrt(len(r)):.2f}) over {len(r)}; "
          f"2014-21 {100*r[g.year.to_numpy() <= 2021].mean():+5.2f}, 2022+ {100*r[g.year.to_numpy() >= 2022].mean():+5.2f}")

print("\n2. held-out brier by season and on teams in their first three seasons after a conference move, vs the EPA refit, 1e-4")
loss = {n: ((d.loc[base.index].team1_win_prob - d.loc[base.index].team1_win) ** 2) for n, d in dfs.items()}
fbs = base.fbs.to_numpy()
for n in names:
    if n == "nc_fit_epa":
        continue
    diff = (loss[n] - loss["nc_fit_epa"])[fbs]
    yrs = base.year[fbs]
    mv = base_mov[fbs]
    per = {y: round(1e4 * diff[yrs == y].mean(), 1) for y in range(2019, 2027)}
    print(f"  {n:>22}: by season {per}")
    for label, m in (("moved 2022+", mv & (yrs >= 2022)), ("not moved 2022+", ~mv & (yrs >= 2022)), ("moved <=2021", mv & (yrs <= 2021)), ("not moved <=2021", ~mv & (yrs <= 2021))):
        x = diff[m]
        print(f"      {label:>16}: {1e4*x.mean():+6.2f} over {m.sum():5d}  t {x.mean()/(x.std()/np.sqrt(len(x))):+5.2f}")
