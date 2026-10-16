"""Where the saturating step gains, whether it removes the post-mismatch bias, and M from the margins alone."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent))
from mismatch_efficiency import OUT, PTS, team_games  # noqa: E402

name, control = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("cm_sat_m90", "cm_sat_ctl")
a = pd.read_parquet(OUT / f"{name}.parquet").drop_duplicates("game_id").set_index("game_id")
b = pd.read_parquet(OUT / f"{control}.parquet").drop_duplicates("game_id").set_index("game_id")
b = b.loc[a.index]
diff = ((a.team1_win_prob - a.team1_win) ** 2 - (b.team1_win_prob - b.team1_win) ** 2)

# Per game: was either team coming off a lopsided game (|expected| >= 28 under the control), and its game number.
tg = team_games(b.reset_index())
prev_lop = tg.assign(lop=tg.expected.abs() >= 28).sort_values(["team", "year", "date"])
prev_lop["prev_lop"] = prev_lop.groupby(["team", "year"]).lop.shift(1).fillna(False).astype(bool)
prev_lop["lop_so_far"] = prev_lop.groupby(["team", "year"]).lop.cumsum() - prev_lop.lop
by_game = prev_lop.groupby("game_id").agg(prev_lop=("prev_lop", "max"), g=("g", "min"),
                                          lop_so_far=("lop_so_far", "max"))
p = b.team1_win_prob.clip(1e-9, 1 - 1e-9)
frame = pd.DataFrame({"diff": diff, "year": a.year, "fbs": a.fbs,
                      "expected": (PTS * np.log(p / (1 - p))).abs()}).join(by_game)

def line(label, rows):
    d = frame.loc[rows, "diff"]
    t = d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 else float("nan")
    print(f"  {label:58s} n {len(d):5d}  {d.mean() * 1e4:+6.2f}e-4 (t {t:+4.1f})")

print(f"{name} vs {control}, brier change per game")
for window, w in (("2014-21", frame.year.between(2014, 2021)), ("2022-26", frame.year >= 2022), ("2014-26", frame.year >= 2014)):
    print(f" {window}")
    fbs = frame.fbs & w
    line("FBS v FBS, all", fbs)
    line("FBS v FBS, either team coming off a lopsided game", fbs & frame.prev_lop)
    line("FBS v FBS, either team had 1+ lopsided game this season", fbs & (frame.lop_so_far >= 1))
    line("FBS v FBS, both teams in games 1-5", fbs & (frame.g <= 5))
    line("FBS v FBS, competitive (|expected| < 14) after a lopsided game", fbs & frame.prev_lop & (frame.expected < 14))
    line("every game", w)

# The bias the step was meant to remove: next competitive game's surprise after a lopsided favorite game.
for label, df in (("control", b.reset_index()), (name, a.reset_index())):
    tgx = team_games(df)
    close_next = tgx.next_expected.abs() < 14
    for window, w in (("2002-21", tgx.year <= 2021), ("2022-26", tgx.year >= 2022)):
        after = tgx[w & close_next & (tgx.expected >= 21)].next_surprise
        base = tgx[w & close_next & (tgx.expected.abs() < 14)].next_surprise
        print(f" {label:12s} {window}: next competitive surprise after a 21+ favorite game {after.mean():+.2f} ± {after.std() / np.sqrt(len(after)):.2f}"
              f"  (after a close game {base.mean():+.2f})")

# M from the margins alone: E[margin | mu] = M tanh(mu / M) on the control's games through 2021.
tr = b[b.year <= 2021]
pp = tr.team1_win_prob.clip(1e-9, 1 - 1e-9)
mu = PTS * np.log(pp / (1 - pp))
m = (tr.home_score - tr.away_score).to_numpy()
fit = minimize_scalar(lambda M: np.mean((m - M * np.tanh(mu / M)) ** 2), bounds=(20, 2000), method="bounded")
lin = np.mean((m - mu) ** 2)
print(f" least-squares M on 2002-21 margins: {fit.x:.0f} points (mse {fit.fun:.2f} vs linear {lin:.2f})")
