"""ncaafb: opponent-adjusted success rates by kind and situation, against margins.

The NFL check (`/tmp/passrun/situational_check.py`) used raw season-to-date
rates, which on ncaafb would mostly measure the schedule. Here each cell's
rate is a ridge refit before every week on that season's earlier contests,

    rate[contest] = center + home + O[offense] + D[defense],  weight = downs

with each team's O and D pulled toward `CARRY` of where last season ended,
by `K` downs -- the season's games doing the opponent adjustment, last
season the prior. Every game with play-by-play is in the fit; FBS v FBS
games once both teams have 4 earlier contests are scored.

Margins: OLS on the rates' gaps, fit 2014-2021, scored 2022-2026. Beside the
success model: a logistic on its logit and the gaps, brier, the same split.

    python ncaafb_cells.py [K]
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cassandra.objective import _in_fbs

HOME = Path.home() / ".cassandra/success_rate"
K = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
CARRY = 0.5
MIN_GAMES = 4
TRAIN_THROUGH = 2021

plays = pd.read_parquet(HOME / "mixed/ncaafb_plays.parquet")
seasons = pickle.loads((HOME / "units/ncaafb_seasons.pkl").read_bytes())
rows = []
for s in seasons:
    for i, w in enumerate(s.weeks):
        for g in w.games:
            if g.home_score is None or g.away_score is None:
                continue
            rows.append((g.game_id, s.year, i, g.date, g.home, g.away, g.home_score - g.away_score,
                         bool(g.neutral_site)))
games = pd.DataFrame(rows, columns=["game_id", "season", "week", "date", "home", "away", "margin", "neutral"])
games = games.drop_duplicates("game_id").set_index("game_id")
plays = plays[plays.game_id.isin(games.index)].copy()
plays["week"] = plays.game_id.map(games.week)
plays["offense"] = np.where(plays.offense_is_home, plays.game_id.map(games.home), plays.game_id.map(games.away))
plays["defense"] = np.where(plays.offense_is_home, plays.game_id.map(games.away), plays.game_id.map(games.home))

owp = np.where(plays.offense_is_home, plays.home_wp, 1 - plays.home_wp)
late = plays.period == 4
short = ((plays.down >= 3) & (plays.distance <= 2)) | ((plays.down == 2) & (plays.distance <= 1))
long = ((plays.down >= 3) & (plays.distance >= 7)) | ((plays.down == 2) & (plays.distance >= 12))
two_min = (plays.period == 2) & (plays.clock_seconds <= 120)
plays["situation"] = np.select(
    [late & (owp > 0.8), late & (owp < 0.2), two_min, short, long],
    ["clock", "hurry", "two_min", "short", "long"], "neutral")
sit, dropback = plays.situation, plays.dropback
CELLS = {
    "all": pd.Series(True, index=plays.index),
    "pass": dropback, "run": ~dropback,
    "n_pass": (sit == "neutral") & dropback, "n_run": (sit == "neutral") & ~dropback,
    "op_pass": (sit == "long") & dropback, "or_run": (sit == "short") & ~dropback,
    "clock_run": (sit == "clock") & ~dropback, "hurry_pass": sit.isin(["hurry", "two_min"]) & dropback,
}
t = plays.groupby("situation").agg(share=("win", "size"), dropbacks=("dropback", "mean"))
t["share"] /= len(plays)
t = t.join(plays.groupby(["situation", "dropback"]).win.mean().unstack().rename(columns={True: "SR_pass", False: "SR_run"}))
print(t.round(3).to_string())
print(f"dropback share {plays.dropback.mean():.3f}; runs the text calls scrambles "
      f"{plays.loc[plays.rush_type, 'scramble_text'].mean():.2%}", flush=True)

keys = ["game_id", "season", "week", "offense", "defense", "offense_is_home"]
contest = plays[keys].drop_duplicates().set_index(["game_id", "offense_is_home"])
for name, mask in CELLS.items():
    g = plays[mask].groupby(["game_id", "offense_is_home"]).win.agg(["sum", "size"])
    contest[f"w_{name}"] = g["sum"]
    contest[f"n_{name}"] = g["size"]
contest = contest.fillna({c: 0 for c in contest if c[:2] in ("w_", "n_")}).reset_index()
contest["home_sign"] = np.where(games.neutral.reindex(contest.game_id).to_numpy(), 0.0,
                                np.where(contest.offense_is_home, 1.0, -1.0))


def ridge(rows: pd.DataFrame, cell: str, teams: dict, prior: dict) -> tuple[float, dict, dict]:
    """center, O and D by team, fit on `rows` with each team pulled toward `prior`."""
    n = rows[f"n_{cell}"].to_numpy(float)
    ok = n > 0
    rows, n = rows[ok], n[ok]
    y = rows[f"w_{cell}"].to_numpy() / n
    T = len(teams)
    r = np.arange(len(rows))
    oi = rows.offense.map(teams).to_numpy()
    di = rows.defense.map(teams).to_numpy()
    X = sparse.csr_matrix(
        (np.r_[np.ones(len(rows)), rows.home_sign.to_numpy(), np.ones(len(rows)), np.ones(len(rows))],
         (np.r_[r, r, r, r], np.r_[np.zeros(len(rows), int), np.ones(len(rows), int), 2 + oi, 2 + T + di])),
        shape=(len(rows), 2 + 2 * T))
    A = (X.T @ sparse.diags(n) @ X).toarray()
    b = X.T @ (n * y)
    p0 = np.zeros(2 + 2 * T)
    for team, j in teams.items():
        p0[2 + j], p0[2 + T + j] = prior.get((team, "O"), 0.0), prior.get((team, "D"), 0.0)
    A[2:, 2:] += np.eye(2 * T) * K
    b[2:] += K * p0[2:]
    A[0, 0] += 1e-6
    A[1, 1] += 1e-6
    beta = np.linalg.solve(A, b)
    return beta[0], {t_: beta[2 + j] for t_, j in teams.items()}, {t_: beta[2 + T + j] for t_, j in teams.items()}


features = []
priors = {cell: {} for cell in CELLS}
for season, sc in contest.groupby("season"):
    teams = {t_: j for j, t_ in enumerate(sorted(set(sc.offense) | set(sc.defense)))}
    played = pd.concat([sc[["week", "offense"]].rename(columns={"offense": "team"}),
                        sc[["week", "defense"]].rename(columns={"defense": "team"})]).drop_duplicates()
    season_games = games[games.season == season]
    for week in sorted(sc.week.unique()):
        before = sc[sc.week < week]
        now = season_games[season_games.week == week]
        if before.empty or now.empty:
            continue
        prior_games = played[played.week < week].groupby("team").size() // 1
        counts = before.groupby("offense").game_id.nunique()
        row = pd.DataFrame(index=now.index)
        row["season"], row["hfa"] = season, (~now.neutral).astype(float)
        row["ng_home"] = now.home.map(counts).fillna(0).to_numpy()
        row["ng_away"] = now.away.map(counts).fillna(0).to_numpy()
        for cell in CELLS:
            _, O, D = ridge(before, cell, teams, priors[cell])
            row[f"o_{cell}"] = now.home.map(O).to_numpy() - now.away.map(O).to_numpy()
            row[f"d_{cell}"] = now.away.map(D).to_numpy() - now.home.map(D).to_numpy()
        features.append(row)
    for cell in CELLS:
        _, O, D = ridge(sc, cell, teams, priors[cell])
        priors[cell] = {**{(t_, "O"): CARRY * v for t_, v in O.items()},
                        **{(t_, "D"): CARRY * v for t_, v in D.items()}}
    print("season", season, "done", flush=True)

data = pd.concat(features).join(games[["home", "away", "margin"]])
data["fbs"] = [_in_fbs(h, int(y)) and _in_fbs(a, int(y)) for h, a, y in zip(data.home, data.away, data.season)]
data = data[data.fbs & (data.ng_home >= MIN_GAMES) & (data.ng_away >= MIN_GAMES)].dropna()
share = plays.dropback.mean()
for s_ in ("o", "d"):
    for w, tag in ((share, "mix"), (0.6, "mix60")):
        data[f"{s_}_{tag}"] = w * data[f"{s_}_pass"] + (1 - w) * data[f"{s_}_run"]
train, test = data[data.season <= TRAIN_THROUGH], data[data.season > TRAIN_THROUGH]
print(f"\nK {K}: FBS v FBS games, train {len(train)}, test {len(test)} (2022-2026); dropback share {share:.3f}")


def both(names):
    return ["hfa"] + [f"{s_}_{n}" for n in names for s_ in ("o", "d")]


def ols(cols):
    A = np.column_stack([np.ones(len(train))] + [train[c] for c in cols])
    beta = np.linalg.lstsq(A, train.margin, rcond=None)[0]
    B = np.column_stack([np.ones(len(test))] + [test[c] for c in cols])
    return beta, (test.margin - B @ beta).to_numpy()


models = {
    "pooled (today)": both(["all"]),
    f"fixed mix {share:.2f}": both(["mix"]),
    "fixed mix 0.60": both(["mix60"]),
    "split pass/run": both(["pass", "run"]),
    "neutral only, split": both(["n_pass", "n_run"]),
    "split + 3rd&long pass + short run": both(["pass", "run", "op_pass", "or_run"]),
    "split + clock runs + hurry passes": both(["pass", "run", "clock_run", "hurry_pass"]),
}
_, ref = ols(models["pooled (today)"])
print("margin, held out: MSE (pts^2), change vs pooled, t")
for name, cols in models.items():
    beta, r = ols(cols)
    d = r**2 - ref**2
    tt = d.mean() / (d.std() / np.sqrt(len(d))) if d.std() else 0.0
    print(f"  {name:36s} {np.mean(r**2):7.2f}  {d.mean():+6.2f}  t {tt:+.2f}   "
          + " ".join(f"{c}={b:+.0f}" for c, b in zip(["c"] + cols, beta) if c not in ("c", "hfa")))

stack = pd.read_parquet(HOME / "combine/stack_ncaafb.parquet", columns=["game_id", "team1_win", "team1_win_prob_success"])
s = data.join(stack.set_index("game_id"), how="inner")
s = s[s.margin != 0]
s["lp"] = np.log(s.team1_win_prob_success / (1 - s.team1_win_prob_success))
s["y"] = s.team1_win.astype(float)
str_, ste = s[s.season <= TRAIN_THROUGH], s[s.season > TRAIN_THROUGH]


def logit_fit(cols):
    A = np.column_stack([np.ones(len(str_))] + [str_[c] for c in cols])
    beta = np.zeros(A.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ beta))
        beta += np.linalg.solve((A * (p * (1 - p))[:, None]).T @ A + 1e-6 * np.eye(len(beta)),
                                A.T @ (str_.y - p) - 1e-6 * beta)
    B = np.column_stack([np.ones(len(ste))] + [ste[c] for c in cols])
    return beta, (1 / (1 + np.exp(-B @ beta)) - ste.y).to_numpy() ** 2


_, base = logit_fit(["lp"])
print(f"\nbeside the success model: train {len(str_)}, test {len(ste)}; model alone {base.mean():.5f}; change x1e-4")
for name, cols in {
    "pooled": ["o_all", "d_all"], "fixed mix": ["o_mix", "d_mix"],
    "split pass/run": ["o_pass", "o_run", "d_pass", "d_run"],
    "3rd&long pass + short run": ["o_op_pass", "d_op_pass", "o_or_run", "d_or_run"],
    "clock runs + hurry passes": ["o_clock_run", "d_clock_run", "o_hurry_pass", "d_hurry_pass"],
}.items():
    beta, b = logit_fit(["lp"] + cols)
    d = b - base
    tt = d.mean() / (d.std() / np.sqrt(len(d)))
    print(f"  + {name:28s} {d.mean() * 1e4:+6.1f}  t {tt:+.2f}   " + " ".join(f"{c}={v:+.1f}" for c, v in zip(cols, beta[2:])))
