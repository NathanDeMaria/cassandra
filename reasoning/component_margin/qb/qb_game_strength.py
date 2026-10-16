"""QB-out by the strength of the man out, with per-game identities and pre-game quality.

Replays the production rule (`qb_out_build.out_teams`, ncaafb handover 1)
from the sweep's per-game starters, keeping who was expected and who
started. A quarterback's quality before a game is his EPA per attempt above
the season's average -- last season at any school plus this season to
date -- shrunk toward the newcomer level by 150 attempts. Residuals are the
roster model's (stack_ncaafb.parquet), FBS v FBS, from the flagged side.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

K = 150.0
PTS = 0.1403 * 167.04790817058102 / np.log(10)
games = json.loads((Path.home() / ".cassandra/qb_strength/ncaafb_qb_games.json").read_text())
d = pd.read_parquet(Path.home() / ".cassandra/success_rate/combine/stack_ncaafb.parquet")
d = d.sort_values(["date", "game_id"]).reset_index(drop=True)
p = d.team1_win_prob_published.clip(1e-9, 1 - 1e-9)
d["exp_margin"] = PTS * np.log(p / (1 - p))
d["margin"] = d.home_score - d.away_score
flags = {g: set(t) for g, t in json.loads(
    (Path.home() / ".cassandra/success_rate/pinned_data/ncaafb_qb_out.json").read_text())["games"].items()}

# Season passing averages and each key's season totals (any school), from the sweep.
year_of = dict(zip(d.game_id.astype(str), d.year))
season_tot = defaultdict(lambda: [0, 0.0])
key_season = defaultdict(lambda: [0, 0.0])
for gid, sides in games.items():
    yr = year_of.get(gid)
    if yr is None:
        continue
    for side in sides.values():
        for key, (att, epa) in side["passers"].items():
            season_tot[yr][0] += att
            season_tot[yr][1] += epa
            key_season[(key, yr)][0] += att
            key_season[(key, yr)][1] += epa
avg = {y: e / a for y, (a, e) in season_tot.items()}
# Newcomers: first seasons with 100+ attempts and none the year before.
first = [((e - a * avg[y]) / a, a) for (k, y), (a, e) in key_season.items()
         if a >= 100 and y - 1 in avg and key_season.get((k, y - 1), [0])[0] == 0]
newcomer = sum(q * a for q, a in first) / sum(a for _, a in first)
print(f"newcomer level {newcomer:+.3f} EPA/att over {len(first)} first seasons")

# Walk each team's season in order, the production rule, recording identities.
rows = []
to_date = defaultdict(lambda: [0, 0.0])  # (team, year, key) -> this season so far
for r in d.itertuples():
    gid = str(r.game_id)
    sides = games.get(gid)
    for side, team, opp in (("home", r.home_team, r.away_team), ("away", r.away_team, r.home_team)):
        rows.append(dict(gid=gid, year=r.year, date=r.date, team=team, opp=opp, side=side,
                         fbs=r.fbs, resid=(1 if side == "home" else -1) * (r.margin - r.exp_margin),
                         rating=r.r_h if side == "home" else r.r_a,
                         qb=sides[side] if sides else None,
                         flagged=team in flags.get(gid, set()),
                         opp_flagged=opp in flags.get(gid, set())))
tg = pd.DataFrame(rows)

def quality(key, yr, team):
    a0, e0 = key_season.get((key, yr - 1), (0, 0.0))
    above = e0 - a0 * avg.get(yr - 1, 0.0)
    a1, e1 = to_date[(team, yr, key)]
    above += e1 - a1 * avg[yr]
    return (above + K * newcomer) / (a0 + a1 + K), a0 + a1

starter, starts, first_start, covering, changes = {}, {}, {}, {}, Counter()
out = []
for r in tg.itertuples():
    ts = (r.team, r.year)
    qb = r.qb
    rec = dict(i=r.Index)
    if qb is not None and qb["starter_key"]:
        today = qb["starter_key"]
        tally = starts.setdefault(ts, Counter())
        order = first_start.setdefault(ts, {})
        expected = starter.get(ts)
        standing = lambda k: (tally[k], -order.get(k, len(order)))
        rec["expected"], rec["today"] = expected, today
        rec["q_today"], rec["att_today"] = quality(today, r.year, r.team)
        if expected is not None:
            rec["q_expected"], rec["att_expected"] = quality(expected, r.year, r.team)
            rec["expected_starts"] = tally[expected]
        rec["changes_before"] = changes[ts]
        if expected is None or standing(today) > standing(expected):
            if expected is not None and expected != today:
                changes[ts] += 1
            starter[ts] = today
            covering.pop(ts, None)
            rec["rule_out"] = False
        elif expected in qb["snap_keys"]:
            covering.pop(ts, None)
            rec["rule_out"] = False
        else:
            rec["rule_out"] = True
            cover, streak = covering.get(ts, (None, 0))
            streak = streak + 1 if cover == today else 1
            if streak >= 1:  # ncaafb handover
                starter[ts] = today
                changes[ts] += 1
                covering.pop(ts, None)
        tally[today] += 1
        order.setdefault(today, len(order))
        for key, (att, epa) in qb["passers"].items():
            t = to_date[(r.team, r.year, key)]
            t[0] += att
            t[1] += epa
    out.append(rec)
tg = tg.join(pd.DataFrame(out).set_index("i"))
fbs_mean = tg[tg.fbs].groupby("year").rating.mean()
tg["strength"] = (tg.rating - tg.year.map(fbs_mean)) * 0.1403
agree = tg[tg.flagged | (tg.rule_out == True)]
print(f"flags in the index {tg.flagged.sum()}, rule replayed here {(tg.rule_out == True).sum()}, "
      f"both {(tg.flagged & (tg.rule_out == True)).sum()}")

def ols(df, xs, label):
    X = np.column_stack([np.ones(len(df))] + [df[x].to_numpy(float) for x in xs])
    y = df.resid.to_numpy(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (e @ e) / (len(y) - X.shape[1]))
    print(f"  {label:46s} n {len(y):4d}  " + "  ".join(
        f"{n} {v:+.2f}±{s:.2f}" for n, v, s in zip(["const"] + xs, b, se)))

fl = tg[tg.fbs & tg.flagged & ~tg.opp_flagged & tg.q_expected.notna()].copy()
fl["q_out10"] = (fl.q_expected - newcomer) / 0.1
fl["q_in10"] = (fl.q_today - newcomer) / 0.1
fl["gap10"] = fl.q_out10 - fl.q_in10
fl["str10"] = fl.strength / 10
fl["settled"] = (fl.changes_before == 0).astype(float)
fl["in_better"] = (fl.q_in10 > fl.q_out10).astype(float)
print("\nflagged FBS team-games (residual after the flat 2.13-pt penalty), per 0.1 EPA/att above a newcomer")
for lo, hi in ((2014, 2021), (2022, 2026), (2014, 2026)):
    s = fl[fl.year.between(lo, hi)]
    print(f" {lo}-{hi}")
    ols(s, [], "flat")
    ols(s, ["q_out10"], "quality of the man out")
    ols(s, ["q_out10", "q_in10"], "out and in")
    ols(s, ["gap10"], "out less in")
    ols(s, ["settled"], "no change at QB yet this season")
    ols(s, ["gap10", "settled"], "out less in, and settled")
fl.to_parquet(Path.home() / ".cassandra/qb_strength/flagged.parquet")

# Every flagged team-game the replayed rule agrees on, any tier: the pre-game gap for a knob.
gap_rows = tg[tg.flagged & (tg.rule_out == True) & tg.q_expected.notna() & tg.q_today.notna()]
index = {}
for r in gap_rows.itertuples():
    index.setdefault(r.gid, {})[r.team] = round((r.q_expected - r.q_today) / 0.1, 4)
(Path.home() / ".cassandra/qb_strength/ncaafb_qb_gap.json").write_text(json.dumps(index))
print("gap index:", sum(len(v) for v in index.values()), "team-games")
