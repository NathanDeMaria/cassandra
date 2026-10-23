"""First NFL look: is success rate worth splitting into dropbacks and runs?

1. Reliability of each team-season rate (odd/even games, Spearman-Brown) and
   season-to-season carry, for offense and defense, pass and run.
2. Out of sample: each contest's EPA/play after week 9, predicted from the
   offense's and defense's rates through week 9 -- pooled SR vs the four
   split rates vs split + interactions, and one rate at a fixed mix. Fit
   2006-2017, scored 2018-2025. `nfl_cells.py` asks the same of margins.

    python nfl_split.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cassandra
from cassandra.epa_build import is_down

sys.path.insert(0, str(Path(__file__).parent))
from kinds import kind_of  # noqa: E402

print("cassandra from", cassandra.__file__)
df = pd.read_parquet(
    "/home/dev/.cassandra/success_rate/nfl_plays.parquet",
    columns=["game_id", "season", "week", "offense", "defense", "play_type", "text", "epa"],
)
df = df[df.season <= 2025]
keep = [is_down(t, x) for t, x in zip(df.play_type, df.text)]
df = df[keep].copy()
df["kind"] = [kind_of(t, x) for t, x in zip(df.play_type, df.text)]
df["win"] = (df.epa > 0).astype(float)
print("dropback share", round((df.kind == "pass").mean(), 3), "SR pass", round(df[df.kind == "pass"].win.mean(), 3), "run", round(df[df.kind == "run"].win.mean(), 3))

# --- 1. reliability --------------------------------------------------------
def rates(frame, side):
    g = frame.groupby(["season", side, "kind"]).win.agg(["sum", "count"]).unstack("kind")
    out = pd.DataFrame({k: g["sum"][k] / g["count"][k] for k in ("pass", "run")})
    out["all"] = g["sum"].sum(axis=1) / g["count"].sum(axis=1)
    return out

games = df[["season", "game_id", "week"]].drop_duplicates()
games["half"] = games.groupby("season").week.rank(method="dense").astype(int) % 2
df = df.merge(games[["game_id", "half"]], on="game_id")
print("\nreliability (odd/even games, Spearman-Brown) | season-to-season r | between-team sd of true rate")
for side in ("offense", "defense"):
    a, b = rates(df[df.half == 0], side), rates(df[df.half == 1], side)
    whole = rates(df, side).reset_index()
    nxt = whole.copy()
    nxt["season"] -= 1
    yoy = whole.merge(nxt, on=["season", side], suffixes=("", "_next"))
    for k in ("pass", "run", "all"):
        r = a[k].corr(b[k])
        rel = 2 * r / (1 + r)
        true_sd = np.sqrt(max(rel, 0)) * whole[k].std()
        print(f"  {side:7s} {k:4s} reliability {rel:5.2f}  yoy r {yoy[k].corr(yoy[k + '_next']):5.2f}  true sd {true_sd:.3f}")

# --- 2. out of sample ------------------------------------------------------
early = df[df.week <= 9]
late = df[(df.week > 9) & (df.week <= 18)]
off = rates(early, "offense").add_prefix("o_")
dfn = rates(early, "defense").add_prefix("d_")
contest = late.groupby(["season", "game_id", "offense", "defense"]).epa.agg(["mean", "count"]).reset_index()
contest = contest.join(off, on=["season", "offense"]).join(dfn, on=["season", "defense"]).dropna()
for c in [c for c in contest if c[:2] in ("o_", "d_")]:
    contest[c] -= contest.groupby("season")[c].transform("mean")
contest["pp"] = contest.o_pass * contest.d_pass
contest["rr"] = contest.o_run * contest.d_run
contest["pr"] = contest.o_pass * contest.d_run
contest["rp"] = contest.o_run * contest.d_pass
contest["share_gap"] = (contest.o_pass - contest.o_run) * (contest.d_pass - contest.d_run)

def fit(cols, train, test):
    X = np.column_stack([np.ones(len(train))] + [train[c] for c in cols])
    w = train["count"].to_numpy()
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], train["mean"] * np.sqrt(w), rcond=None)[0]
    Xt = np.column_stack([np.ones(len(test))] + [test[c] for c in cols])
    resid = test["mean"] - Xt @ beta
    return beta, np.average(resid**2, weights=test["count"]), resid

train, test = contest[contest.season <= 2017], contest[contest.season > 2017]
base = np.average((test["mean"] - np.average(train["mean"], weights=train["count"])) ** 2, weights=test["count"])
models = {
    "pooled SR": ["o_all", "d_all"],
    "split SR": ["o_pass", "o_run", "d_pass", "d_run"],
    "split + same-kind products": ["o_pass", "o_run", "d_pass", "d_run", "pp", "rr"],
    "split + all four products": ["o_pass", "o_run", "d_pass", "d_run", "pp", "rr", "pr", "rp"],
    "split + lean-vs-lean": ["o_pass", "o_run", "d_pass", "d_run", "share_gap"],
}
print(f"\ncontests after week 9: train {len(train)}  test {len(test)} (2018-2025); weighted MSE of game EPA/play")
print(f"  intercept only            {base:.5f}")
ref = ref_mse = None
for name, cols in models.items():
    beta, mse, resid = fit(cols, train, test)
    if ref is None:
        ref, ref_mse = resid, mse
    d = (resid**2 - ref**2) * test["count"] / test["count"].mean()
    t = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() else 0.0
    print(f"  {name:28s} {mse:.5f}  R2 {1 - mse / base:6.4f}  vs pooled {mse - ref_mse:+.6f}  t {t:+.2f}")
    print("     coefs", " ".join(f"{c}={b:+.2f}" for c, b in zip(["c"] + cols, beta)))

print("\nsingle rate, dropbacks weighted w (play share is 0.60):")
for w in (0.60, 0.70, 0.75, 0.80, 0.90):
    for s in ("o", "d"):
        contest[f"{s}_mix"] = w * contest[f"{s}_pass"] + (1 - w) * contest[f"{s}_run"]
    beta, mse, resid = fit(["o_mix", "d_mix"], contest[contest.season <= 2017], contest[contest.season > 2017])
    print(f"  w {w:.2f}  {mse:.5f}  R2 {1 - mse / base:6.4f}  vs pooled {mse - ref_mse:+.6f}")
