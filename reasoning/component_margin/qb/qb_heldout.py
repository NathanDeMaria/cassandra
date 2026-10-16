import numpy as np, pandas as pd
from pathlib import Path
PTS = 0.1403 * 167.04790817058102 / np.log(10)
fl = pd.read_parquet(Path.home() / ".cassandra/qb_strength/flagged.parquet")
fl = fl[fl.rule_out == True].copy()
print("flags where the replayed rule agrees:", len(fl))
def ols(df, xs):
    X = np.column_stack([np.ones(len(df))] + [df[x].to_numpy(float) for x in xs])
    y = df.resid.to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / (len(y) - X.shape[1]))
    return b, se
for lo, hi in ((2014, 2021), (2022, 2026), (2014, 2026)):
    s = fl[fl.year.between(lo, hi)]
    b, se = ols(s, ["gap10"])
    b2, se2 = ols(s, ["q_out10"])
    print(f" {lo}-{hi} n {len(s)}  gap: const {b[0]:+.2f}±{se[0]:.2f} slope {b[1]:+.2f}±{se[1]:.2f}   "
          f"out only: slope {b2[1]:+.2f}±{se2[1]:.2f}   mean gap10 {s.gap10.mean():+.2f} sd {s.gap10.std():.2f}")
# Held out: slope fit 2014-21, applied to 2022-26, with and without re-levelling.
tr, te = fl[fl.year <= 2021], fl[fl.year >= 2022]
exp_side = te.resid.to_numpy() * 0  # rebuild the model's margin from the stack
stack = pd.read_parquet(Path.home() / ".cassandra/success_rate/combine/stack_ncaafb.parquet")
p = stack.team1_win_prob_published.clip(1e-9, 1 - 1e-9)
em = dict(zip(stack.game_id.astype(str), PTS * np.log(p / (1 - p))))
sign = np.where(te.side == "home", 1.0, -1.0)
exp_side = sign * te.gid.map(em).to_numpy()
won = (te.resid.to_numpy() + exp_side > 0).astype(float)
base = 1 / (1 + np.exp(-exp_side / PTS))
b, _ = ols(tr, ["gap10"])
for label, corr in (("slope only, centred on train", b[1] * (te.gap10 - tr.gap10.mean()).to_numpy()),
                    ("slope and train intercept", b[0] + b[1] * te.gap10.to_numpy()),
                    ("slope only, uncentred (k * gap)", b[1] * te.gap10.to_numpy()),
                    ("fixed 1.5 per 0.1, uncentred", -1.5 * te.gap10.to_numpy())):
    new = 1 / (1 + np.exp(-(exp_side + corr) / PTS))
    diff = (new - won) ** 2 - (base - won) ** 2
    t = diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))
    by = pd.Series(diff).groupby(te.year.to_numpy()).mean()
    print(f"  held out {label:34s} brier on flagged {diff.mean()*1e4:+6.1f}e-4 (t {t:+.2f}), "
          f"seasons better {(by < 0).sum()}/{len(by)}; over all FBS games {diff.sum()/3503*1e4:+.2f}e-4")
