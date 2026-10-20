"""Which parts of a margin last: efficiency, explosiveness, or the rest?

    python residual.py ncaafb|nfl

A game's margin, home less away, splits by the offenses' EPA totals:

    margin = k * (efficiency margin + explosiveness margin) + residual

with `k` points per point of EPA margin (fit over every game; the clip
puts it above 1), the efficiency margin each offense's success-implied EPA
times its snaps (`persistence.py`'s split), and the residual what the
offenses' plays don't account for -- special teams, the clip, the expected
points model's own error. Turnovers sit inside the EPA, so their bounces
are not in the residual.

Each part, in points, gets one rating per team per season
(`persistence.rate_margins`), and the questions of `persistence.py`: how
reliable a season of it is, how much carries into the next season, and,
from week five on, whether rating the parts apart -- each shrunk by its own
measured noise -- predicts next week's margin better than rating the
margin whole, which is what the record does.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from persistence import (  # noqa: E402
    FIRST_WEEK,
    MIN_HALF,
    MIN_SEASON,
    TRAIN_THROUGH,
    load,
    rate_margins,
)

PARTS = ("margin", "eff", "expl", "rest")
SPLITS = 8


def components(games: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """One row per game with play-by-play: the margin and its three parts, in points."""
    # `load` writes each game's home offense first and its away offense second.
    rows = plays.groupby("game_id")
    first = rows.nth(0).set_index("game_id")
    second = rows.nth(1).set_index("game_id").loc[first.index]
    df = games.set_index("game_id").loc[first.index]
    assert (first.offense == df.home_team).all(), "first row is not the home offense"
    epa = first.epa * first.snaps - second.epa * second.snaps
    eff = first.eff * first.snaps - second.eff * second.snaps
    design = np.c_[np.ones(len(df)), df.home, epa]
    _, _, k = np.linalg.lstsq(design, df.margin, rcond=None)[0]
    out = df[["year", "week", "home_team", "away_team", "home", "fbs", "team1_win_prob"]].copy()
    out["margin"] = df.margin
    out["eff"] = k * eff
    out["expl"] = k * (epa - eff)
    out["rest"] = df.margin - k * epa
    print(f"points per point of EPA margin k = {k:.3f};"
          f" game-level sd, points: margin {out.margin.std():.1f}, efficiency {out.eff.std():.1f},"
          f" explosiveness {out.expl.std():.1f}, rest {out.rest.std():.1f}")
    return out.reset_index()


def as_margin(df: pd.DataFrame, part: str) -> pd.DataFrame:
    return df.assign(margin=df[part])


def reliability(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    measured = {}
    sides = pd.concat([df[["year", "home_team"]].set_axis(["year", "team"], axis=1),
                       df[["year", "away_team"]].set_axis(["year", "team"], axis=1)])
    games_per_half = sides.groupby(["year", "team"]).size().median() / 2
    full = {}
    for year, season in df.groupby("year"):
        for part in PARTS:
            full[year, part] = rate_margins(as_margin(season, part), 0.5)
    print(f"\n{'part':6s} {'half r':>7s} {'season rel':>10s} {'true sd':>8s} {'noise/game':>10s}"
          f" {'carryover':>9s} {'r(t,t+1)':>9s} {'prior, games':>12s}")
    for part in PARTS:
        a_all, b_all, years = [], [], []
        for year, season in df.groupby("year"):
            ids = season.game_id.unique()
            for _ in range(SPLITS):
                half = set(rng.choice(ids, len(ids) // 2, replace=False))
                ra, na = rate_margins(as_margin(season[season.game_id.isin(half)], part), 0.5)
                rb, nb = rate_margins(as_margin(season[~season.game_id.isin(half)], part), 0.5)
                for t in ra:
                    if na.get(t, 0) >= MIN_HALF and nb.get(t, 0) >= MIN_HALF and t in rb:
                        a_all.append(ra[t])
                        b_all.append(rb[t])
                        years.append(year)
        h = pd.DataFrame({"year": years, "a": a_all, "b": b_all})
        a = h.a - h.groupby("year").a.transform("mean")
        b = h.b - h.groupby("year").b.transform("mean")
        r = float(np.corrcoef(a, b)[0, 1])
        true_var = float(np.mean(a * b))
        noise = float((np.mean(a * a) + np.mean(b * b)) / 2 - true_var) * games_per_half
        pairs = []
        yrs = sorted(df.year.unique())
        for y0, y1 in zip(yrs, yrs[1:]):
            (r0, n0), (r1, n1) = full[y0, part], full[y1, part]
            for t in r0:
                if n0.get(t, 0) >= MIN_SEASON and n1.get(t, 0) >= MIN_SEASON and t in r1:
                    pairs.append((y0, r0[t], r1[t]))
        s = pd.DataFrame(pairs, columns=["year", "a", "b"])
        sa = s.a - s.groupby("year").a.transform("mean")
        sb = s.b - s.groupby("year").b.transform("mean")
        carry = float(np.mean(sa * sb)) / true_var
        measured[part] = noise / true_var
        print(f"{part:6s} {r:7.3f} {2 * r / (1 + r):10.3f} {np.sqrt(true_var):8.2f}"
              f" {np.sqrt(noise):10.2f} {carry:9.3f} {float(np.corrcoef(sa, sb)[0, 1]):9.3f}"
              f" {noise / true_var:12.1f}")
    return measured


def features(df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    out = []
    for year, season in df.groupby("year"):
        for week in sorted(season.week.unique()):
            if week < FIRST_WEEK:
                continue
            before = season[season.week < week]
            if before.empty:
                continue
            fits = {part: rate_margins(as_margin(before, part), priors[part]) for part in PARTS}
            seen = fits["margin"][1]
            for g in season[season.week == week].itertuples():
                h, a = g.home_team, g.away_team
                if seen.get(h, 0) < 3 or seen.get(a, 0) < 3:
                    continue
                row = {"year": year, "fbs": g.fbs, "home": g.home, "margin": g.margin,
                       "win": float(g.margin > 0), "tie": g.margin == 0,
                       "prob": g.team1_win_prob}
                for part, (r, _) in fits.items():
                    row[f"g_{part}"] = r.get(h, 0.0) - r.get(a, 0.0)
                out.append(row)
    return pd.DataFrame(out)


def predict(f: pd.DataFrame) -> None:
    f = f[f.fbs].copy()
    eps = 1e-6
    f["prod"] = np.log((f.prob + eps) / (1 - f.prob + eps))
    train, test = f[f.year <= TRAIN_THROUGH], f[f.year > TRAIN_THROUGH]
    print(f"\nnext-week margins, week {FIRST_WEEK}+ : {len(train)} games through"
          f" {TRAIN_THROUGH}, {len(test)} after")
    specs = {
        "margin, rated whole": ["home", "g_margin"],
        "parts, each rated apart": ["home", "g_eff", "g_expl", "g_rest"],
        "efficiency + explosiveness": ["home", "g_eff", "g_expl"],
        "efficiency alone": ["home", "g_eff"],
        "margin whole + parts": ["home", "g_margin", "g_eff", "g_expl", "g_rest"],
        "published": ["prod"],
        "published + parts": ["prod", "g_eff", "g_expl", "g_rest"],
    }
    print(f"{'features':26s} {'test RMSE':>9s} {'test brier':>10s}   coefficients on the gaps")
    losses = {}
    for name, cols in specs.items():
        ols = LinearRegression().fit(train[cols], train.margin)
        rmse = float(np.sqrt(np.mean((ols.predict(test[cols]) - test.margin) ** 2)))
        decided = train[~train.tie]
        logit = LogisticRegression(C=1e6, max_iter=2000).fit(decided[cols], decided.win)
        losses[name] = (logit.predict_proba(test[cols])[:, 1] - test.win.to_numpy()) ** 2
        brier = float(losses[name].mean())
        design = np.c_[np.ones(len(train)), train[cols]]
        resid = train.margin - ols.predict(train[cols])
        cov = np.linalg.inv(design.T @ design) * resid.var()
        coefs = "  ".join(f"{c} {b:+.2f} (t {b / np.sqrt(cov[i + 1, i + 1]):+.1f})"
                          for i, (c, b) in enumerate(zip(cols, ols.coef_)) if c != "home")
        print(f"{name:26s} {rmse:9.3f} {brier:10.5f}   {coefs}")
    print("\nheld-out brier, paired:")
    for a, b in (("parts, each rated apart", "margin, rated whole"),
                 ("margin whole + parts", "margin, rated whole"),
                 ("published + parts", "published")):
        diff = losses[a] - losses[b]
        t = diff.mean() / diff.std() * np.sqrt(len(diff))
        print(f"  {a} - {b}: {diff.mean() * 1e4:+.2f}e-4 (t {t:+.2f})")


def main() -> None:
    league = sys.argv[1]
    rng = np.random.default_rng(1)
    games, plays = load(league)
    df = components(games, plays)
    priors = reliability(df, rng)
    predict(features(df, priors))


if __name__ == "__main__":
    main()
