"""How the read per-play gap moves with what the parts expected, by size of the expected gap."""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import run as R

knobs = json.loads((Path(__file__).parent / "logs/s1.best.json").read_text())["knobs"]
knobs["defense_success_weight"] = knobs["success_weight"]
named, _, index = R._inputs()
warm, early = R._warm()
from cassandra.predictor.opponent_prior import OpponentPriorManager
p = R.ComponentMarginPredictor.from_state_dict({**warm.state_dict(), **R.START, **knobs,
    "opponent_prior_manager": OpponentPriorManager("ncaafb", path=R.PRIORS), "game_epa": index})
p._season = warm._season
p._qb_pending = set(warm._qb_pending)
df = R._replay(p, [s for s in named if 2014 <= s.year])
c = pd.DataFrame.from_dict(p.contests, orient="index", columns=["expected", "read", "level_pts"])
df = df.set_index("game_id").join(c, how="inner")
df["margin"] = df.home_score - df.away_score
for label, rows in (("all read games", df.index == df.index), ("FBS v FBS", df.fbs), ("not FBS v FBS", ~df.fbs)):
    s = df[rows]
    b = np.polyfit(s.expected, s.read, 1)
    print(f"{label:16s} n {len(s):5d}  read gap on expected: slope {b[0]:.3f} intercept {b[1]:+.4f}")
df["bin"] = pd.cut(df.expected.abs(), [0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.7, 3])
sign = np.sign(df.expected)
print(df.assign(exp_abs=df.expected.abs(), read_signed=df.read * sign, margin_signed=df.margin * sign,
                pred_pts=(df.expected.abs() * R.START["points_per_epa"]))
      .groupby("bin", observed=True)[["exp_abs", "read_signed", "pred_pts", "margin_signed"]]
      .agg(["mean"]).round(3).assign(n=df.groupby("bin", observed=True).size()))
for yr in (2014, 2016, 2018, 2020, 2021, 2022, 2023, 2024, 2025):
    s = df[(df.year == yr) & df.fbs]
    print(yr, "FBS slope", round(np.polyfit(s.expected, s.read, 1)[0], 3), "n", len(s))
