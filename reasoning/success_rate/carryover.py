import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from success import load_plays, team_games, team_seasons

g = team_games(load_plays())
lg = g.groupby("season")[["n", "k"]].sum(); lg = lg.k / lg.n
for side in ("offense", "defense"):
    ts = team_seasons(g[g.season <= 2025], side)
    ts["r"] = ts.k / ts.n - lg.reindex(ts.season).to_numpy()
    nxt = ts.assign(season=ts.season - 1)[["season", "team", "r"]]
    j = ts.merge(nxt, on=["season", "team"], suffixes=("", "_next"))
    slope = np.polyfit(j.r, j.r_next, 1)[0]
    print(f"{side}: year-to-year r={j.r.corr(j.r_next):.2f}, slope={slope:.2f}, n={len(j)}")
