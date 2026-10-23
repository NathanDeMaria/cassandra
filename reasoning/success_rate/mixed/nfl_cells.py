"""NFL: do success rates by kind and situation predict margins beyond the pooled rate?

Raw season-to-date rates (no opponent adjustment; NFL schedules are close to
balanced), each shrunk toward the season's cell mean by K downs, for every
game once both teams have 4 earlier games. Fit 2006-2017, scored 2018-2025,
on the final margin. Then the same rates beside the success model's win
probability (`combine/stack_nfl.parquet`), logistic, on brier. Last, each
team's own dropback share beside them.

    python nfl_cells.py [K]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nfl_situations import load  # noqa: E402

K = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
df = load()
sit, kind = df.situation, df.kind
cells = {
    "n_pass": (sit == "neutral") & (kind == "pass"),
    "n_run": (sit == "neutral") & (kind == "run"),
    "op_pass": (sit == "long") & (kind == "pass"),
    "op_run": (sit == "long") & (kind == "run"),
    "or_run": (sit == "short") & (kind == "run"),
    "or_pass": (sit == "short") & (kind == "pass"),
    "clock_run": (sit == "clock") & (kind == "run"),
    "hurry_pass": sit.isin(["hurry", "two_min"]) & (kind == "pass"),
    "pass": kind == "pass",
    "run": kind == "run",
    "all": pd.Series(True, index=df.index),
}

games = df.groupby("game_id").agg(
    season=("season", "first"), date=("date", "first"), hs=("home_score", "first"),
    as_=("away_score", "first"), neutral=("neutral_site", "first"),
)
home = df[df.offense_is_home].groupby("game_id").offense.first()
away = df[~df.offense_is_home].groupby("game_id").offense.first()
games = games.join(home.rename("home")).join(away.rename("away")).dropna()
games["margin"] = games.hs - games.as_


def to_date(side):
    """Each team's rates in each cell over its earlier games this season."""
    parts = []
    for name, mask in cells.items():
        g = df[mask].groupby(["season", "game_id", side]).win.agg(["sum", "count"])
        g.columns = [f"w_{name}", f"k_{name}"]
        parts.append(g)
    per = pd.concat(parts, axis=1).fillna(0).reset_index().rename(columns={side: "team"})
    per = per.merge(games[["date"]], left_on="game_id", right_index=True).sort_values("date")
    cum = per.groupby(["season", "team"])[[c for c in per if c[:2] in ("w_", "k_")]].transform(
        lambda s: s.cumsum().shift(1, fill_value=0)
    )
    cum["games"] = per.groupby(["season", "team"]).cumcount()
    out = pd.concat([per[["season", "game_id", "team"]], cum], axis=1)
    for name in cells:
        mean = df[cells[name]].groupby("season").win.mean()
        m = out.season.map(mean)
        out[name] = (out[f"w_{name}"] + K * m) / (out[f"k_{name}"] + K)
    return out.set_index(["game_id", "team"])[list(cells) + ["games"]]


off, dfn = to_date("offense"), to_date("defense")


def side(frame, team_col):
    keys = pd.MultiIndex.from_arrays([games.index, games[team_col]])
    return frame.reindex(keys).set_axis(games.index)


ho, ao, hd, ad = side(off, "home"), side(off, "away"), side(dfn, "home"), side(dfn, "away")
X = pd.DataFrame(index=games.index)
for name in cells:
    X[f"o_{name}"] = ho[name] - ao[name]  # home offense better
    X[f"d_{name}"] = ad[name] - hd[name]  # home defense allows less
X["hfa"] = (~games.neutral.astype(bool)).astype(float)
ok = (ho.games >= 4) & (ao.games >= 4) & (hd.games >= 4) & (ad.games >= 4)
data = games[ok].join(X[ok]).dropna()
mix = lambda frame, a, b, w: w * frame[a] + (1 - w) * frame[b]
for s in ("o", "d"):
    data[f"{s}_mix"] = mix(data, f"{s}_pass", f"{s}_run", 0.65)
    data[f"{s}_nmix"] = mix(data, f"{s}_n_pass", f"{s}_n_run", 0.65)
train, test = data[data.season <= 2017], data[data.season > 2017]
print(f"K {K}: games train {len(train)}, test {len(test)} (2018-2025)")


def both(names):
    return ["hfa"] + [f"{s}_{n}" for n in names for s in ("o", "d")]


models = {
    "pooled SR (today)": both(["all"]),
    "fixed mix 0.65 pass": both(["mix"]),
    "split pass/run": both(["pass", "run"]),
    "neutral only, split": both(["n_pass", "n_run"]),
    "neutral mix 0.65": both(["nmix"]),
    "neutral split + 3rd&long pass + short run": both(["n_pass", "n_run", "op_pass", "or_run"]),
    "  + contrarian calls": both(["n_pass", "n_run", "op_pass", "or_run", "op_run", "or_pass"]),
    "  + clock runs, hurry passes": both(["n_pass", "n_run", "op_pass", "or_run", "clock_run", "hurry_pass"]),
    "split pass/run + 3rd&long pass + short run": both(["pass", "run", "op_pass", "or_run"]),
}


def ols(cols):
    A = np.column_stack([np.ones(len(train))] + [train[c] for c in cols])
    beta = np.linalg.lstsq(A, train.margin, rcond=None)[0]
    B = np.column_stack([np.ones(len(test))] + [test[c] for c in cols])
    return beta, (test.margin - B @ beta).to_numpy()


_, ref = ols(models["pooled SR (today)"])
print(f"\nmargin, out of sample: MSE (pts^2) and change vs pooled, t on per-game differences")
for name, cols in models.items():
    beta, r = ols(cols)
    d = r**2 - ref**2
    t = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() else 0.0
    print(f"  {name:45s} {np.mean(r**2):7.2f}  {d.mean():+6.2f}  t {t:+.2f}")
    if name.startswith(("neutral split +", "  +", "split pass/run +")):
        print("      ", " ".join(f"{c}={b:+.0f}" for c, b in zip(["c"] + cols, beta) if c != "hfa"))

# --- beside the published success model ----------------------------------
stack = pd.read_parquet("/home/dev/.cassandra/success_rate/combine/stack_nfl.parquet", columns=["game_id", "team1_win", "team1_win_prob_success"])
s = data.join(stack.set_index("game_id"), how="inner")
s = s[s.margin != 0]
s["lp"] = np.log(s.team1_win_prob_success / (1 - s.team1_win_prob_success))
s["y"] = s.team1_win.astype(float)
str_, ste = s[s.season <= 2017], s[s.season > 2017]


def logit_fit(cols, ridge=1e-6):
    A = np.column_stack([np.ones(len(str_))] + [str_[c] for c in cols])
    beta = np.zeros(A.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ beta))
        g = A.T @ (str_.y - p) - ridge * beta
        H = (A * (p * (1 - p))[:, None]).T @ A + ridge * np.eye(len(beta))
        beta += np.linalg.solve(H, g)
    B = np.column_stack([np.ones(len(ste))] + [ste[c] for c in cols])
    p = 1 / (1 + np.exp(-B @ beta))
    return beta, (p - ste.y).to_numpy() ** 2


print(f"\nbeside the success model's win prob: games train {len(str_)}, test {len(ste)}; brier change x1e-4 vs recalibrated model alone")
_, base = logit_fit(["lp"])
print(f"  model alone, brier {base.mean():.5f}")
extra = {
    "pooled SR": ["o_all", "d_all"],
    "fixed mix 0.65": ["o_mix", "d_mix"],
    "split pass/run": ["o_pass", "o_run", "d_pass", "d_run"],
    "split + 3rd&long pass + short run": ["o_pass", "o_run", "d_pass", "d_run", "o_op_pass", "d_op_pass", "o_or_run", "d_or_run"],
    "neutral mix": ["o_nmix", "d_nmix"],
    "neutral split": ["o_n_pass", "o_n_run", "d_n_pass", "d_n_run"],
    "3rd&long pass + short run": ["o_op_pass", "d_op_pass", "o_or_run", "d_or_run"],
    "clock runs + hurry passes": ["o_clock_run", "d_clock_run", "o_hurry_pass", "d_hurry_pass"],
}
for name, cols in extra.items():
    beta, b = logit_fit(["lp"] + cols)
    d = b - base
    t = d.mean() / (d.std() / np.sqrt(len(d)))
    print(f"  + {name:28s} {d.mean() * 1e4:+6.1f}  t {t:+.2f}   " + " ".join(f"{c}={v:+.1f}" for c, v in zip(cols, beta[2:])))

# --- is it the team's own pass share? -------------------------------------
def share_frame(side):
    """Each team's season-to-date dropback share, over every down and over neutral downs."""
    parts = []
    for name in ("pass", "all", "n_pass", "n_run"):
        g = df[cells[name]].groupby(["season", "game_id", side]).win.count().rename(name)
        parts.append(g)
    per = pd.concat(parts, axis=1).fillna(0).reset_index().rename(columns={side: "team"})
    per = per.merge(games[["date"]], left_on="game_id", right_index=True).sort_values("date")
    cum = per.groupby(["season", "team"])[["pass", "all", "n_pass", "n_run"]].transform(lambda x: x.cumsum().shift(1, fill_value=0))
    out = per[["game_id", "team"]].copy()
    out["share"] = cum["pass"] / cum["all"]
    out["nshare"] = cum["n_pass"] / (cum["n_pass"] + cum["n_run"])
    return out.set_index(["game_id", "team"])

so, sd_ = share_frame("offense"), share_frame("defense")
hs_o, as_o, hs_d, as_d = side(so, "home"), side(so, "away"), side(sd_, "home"), side(sd_, "away")
for c in ("share", "nshare"):
    data[f"o_{c}"] = (hs_o[c] - as_o[c]).reindex(data.index)
    data[f"d_{c}"] = (as_d[c] - hs_d[c]).reindex(data.index)
train, test = data[data.season <= 2017], data[data.season > 2017]
print("\nadding each team's own dropback share (season to date):")
more = {
    "pooled SR (today)": both(["all"]),
    "split pass/run + overall share": both(["pass", "run", "share"]),
    "split pass/run + neutral share": both(["pass", "run", "nshare"]),
    "neutral split + neutral share": both(["n_pass", "n_run", "nshare"]),
    "pooled SR + overall share": both(["all", "share"]),
    "shares alone": both(["share", "nshare"]),
}
for name, cols in more.items():
    beta, r = ols(cols)
    d = r**2 - ref**2
    t = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() else 0.0
    print(f"  {name:45s} {np.mean(r**2):7.2f}  {d.mean():+6.2f}  t {t:+.2f}   " + " ".join(f"{c}={b:+.0f}" for c, b in zip(["c"] + cols, beta) if c not in ("c", "hfa")))
