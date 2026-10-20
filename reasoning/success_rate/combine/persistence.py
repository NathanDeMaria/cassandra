"""Is explosiveness a signal of its own, or noise around efficiency?

    python persistence.py ncaafb|nfl

EPA per play splits into what an offense's success rate implies and the
rest,

    epa = alpha + beta * success + explosiveness

fit over every offense-game by snaps, so the two parts are uncorrelated game
by game. Each part is opponent-adjusted within a season by weighted least
squares, `reading = mu + home + O[offense] - D[defense]` by snaps, and then
asked three things:

1. Reliability. Fit on a random half of a season's games and on the other
   half; over teams, the two fits' correlation, stepped up to a whole season
   (Spearman-Brown), and the square root of their covariance -- the spread
   of the true values, since the halves' noise is independent.
2. Carryover. One season's fit against the next's, as a share of the true
   variance: the AR(1) coefficient each part's offseason should regress by.
3. Prediction. From a season's fifth week on, each part's season-to-date
   fit (shrunk by its own measured noise-to-signal ratio) and a rating on
   margins, regressed onto that week's margins through 2021 and scored on
   2022 on -- with the parts apart, and summed back into EPA, so the
   difference is what separate shrinkage is worth.

The games and the published model's win probabilities come from the
harness's replays (`ncaafb_out/published_p`, `out/base_p`), the readings
from the swept index; nothing here reads the shared predictor data.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

DATA = Path.home() / ".cassandra/success_rate/units"
GAMES = {"ncaafb": DATA / "ncaafb_out/published_p.parquet", "nfl": DATA / "out/base_p.parquet"}
INDEX = {"ncaafb": DATA / "ncaafb_index.pkl", "nfl": DATA / "nfl_index.pkl"}
TRAIN_THROUGH = 2021
PARTS = ("epa", "eff", "expl")
#: Plays of prior weight for the reliability fits: enough to identify a team
#: seen once, too little to shrink a team seen all season.
IDENTIFY = 20.0
#: Games a team needs in each half, and in each season, to be compared.
MIN_HALF, MIN_SEASON = 4, 8
SPLITS = 8
FIRST_WEEK = 4


def load(league: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_parquet(GAMES[league]).drop_duplicates("game_id")
    if "fbs" not in games:
        games["fbs"] = True
    games["margin"] = (games.home_score - games.away_score).astype(float)
    games["home"] = np.where(games.neutral_site, 0.0, 1.0)
    # A week counted from the season's first game: ESPN restarts its
    # numbering for the postseason.
    start = games.groupby("year").date.transform("min")
    games["week"] = (games.date - start).dt.days // 7
    index = pickle.loads(INDEX[league].read_bytes())
    rows = []
    for g in games.itertuples():
        e = index.get(str(g.game_id))
        if e is None or not e.home_plays or not e.away_plays:
            continue
        if e.home_success is None or e.away_success is None:
            continue
        rows.append((g.game_id, g.year, g.week, g.home_team, g.away_team, g.home,
                     e.home, e.home_success, e.home_plays))
        rows.append((g.game_id, g.year, g.week, g.away_team, g.home_team, -g.home,
                     e.away, e.away_success, e.away_plays))
    plays = pd.DataFrame(rows, columns=["game_id", "year", "week", "offense", "defense",
                                        "home", "epa", "success", "snaps"])
    w = np.sqrt(plays.snaps.to_numpy())
    design = np.c_[np.ones(len(plays)), plays.success]
    alpha, beta = np.linalg.lstsq(design * w[:, None], plays.epa * w, rcond=None)[0]
    plays["eff"] = alpha + beta * plays.success
    plays["expl"] = plays.epa - plays.eff
    print(f"{league}: {len(plays)} offense-games, epa = {alpha:.3f} + {beta:.3f} * success"
          f" + explosiveness; game-level sd eff {plays.eff.std():.3f},"
          f" expl {plays.expl.std():.3f} EPA/play")
    return games, plays


def fit(df: pd.DataFrame, value: str, weight: str, prior: float) -> tuple[dict, dict, dict]:
    """O and D per team by weighted least squares, `prior` units of weight toward 0."""
    teams = sorted(set(df.offense) | set(df.defense))
    at = {t: i for i, t in enumerate(teams)}
    k, n = len(teams), len(df)
    design = np.zeros((n, 2 + 2 * k))
    design[:, 0] = 1.0
    design[:, 1] = df.home.to_numpy()
    rows = np.arange(n)
    design[rows, 2 + df.offense.map(at).to_numpy()] = 1.0
    design[rows, 2 + k + df.defense.map(at).to_numpy()] = -1.0
    weighted = design.T * df[weight].to_numpy()
    normal = weighted @ design
    normal[2:, 2:] += prior * np.eye(2 * k)
    solved = np.linalg.solve(normal, weighted @ df[value].to_numpy())
    games = df.offense.value_counts().to_dict()
    return dict(zip(teams, solved[2:2 + k])), dict(zip(teams, solved[2 + k:])), games


def rate_margins(games: pd.DataFrame, prior: float) -> tuple[dict, dict]:
    """One rating per team on margins, `margin = home + R[home] - R[away]`, `prior` games toward 0."""
    teams = sorted(set(games.home_team) | set(games.away_team))
    at = {t: i for i, t in enumerate(teams)}
    k, n = len(teams), len(games)
    design = np.zeros((n, 1 + k))
    design[:, 0] = games.home.to_numpy()
    rows = np.arange(n)
    design[rows, 1 + games.home_team.map(at).to_numpy()] += 1.0
    design[rows, 1 + games.away_team.map(at).to_numpy()] -= 1.0
    normal = design.T @ design
    normal[1:, 1:] += prior * np.eye(k)
    normal[0, 0] += 1e-6
    solved = np.linalg.solve(normal, design.T @ games.margin.to_numpy())
    counts = pd.concat([games.home_team, games.away_team]).value_counts().to_dict()
    return dict(zip(teams, solved[1:])), counts


def reliability(plays: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Each part's split-half pairs, offense and defense, pooled over seasons."""
    pairs = []
    for year, season in plays.groupby("year"):
        ids = season.game_id.unique()
        for _ in range(SPLITS):
            half = set(rng.choice(ids, len(ids) // 2, replace=False))
            a, b = season[season.game_id.isin(half)], season[~season.game_id.isin(half)]
            for part in PARTS:
                oa, da, na = fit(a, part, "snaps", IDENTIFY)
                ob, db, nb = fit(b, part, "snaps", IDENTIFY)
                both = [t for t in oa if na.get(t, 0) >= MIN_HALF and nb.get(t, 0) >= MIN_HALF
                        and t in ob]
                for t in both:
                    pairs.append((year, part, "offense", oa[t], ob[t]))
                    pairs.append((year, part, "defense", da[t], db[t]))
    return pd.DataFrame(pairs, columns=["year", "part", "side", "a", "b"])


def carryover(plays: pd.DataFrame) -> pd.DataFrame:
    fits = {}
    for year, season in plays.groupby("year"):
        for part in PARTS:
            o, d, n = fit(season, part, "snaps", IDENTIFY)
            fits[year, part] = (o, d, n)
    pairs = []
    years = sorted(plays.year.unique())
    for year, nxt in zip(years, years[1:]):
        for part in PARTS:
            o0, d0, n0 = fits[year, part]
            o1, d1, n1 = fits[nxt, part]
            for t in o0:
                if n0.get(t, 0) >= MIN_SEASON and n1.get(t, 0) >= MIN_SEASON and t in o1:
                    pairs.append((year, part, "offense", o0[t], o1[t]))
                    pairs.append((year, part, "defense", d0[t], d1[t]))
    return pd.DataFrame(pairs, columns=["year", "part", "side", "a", "b"])


def summarize(halves: pd.DataFrame, seasons: pd.DataFrame, plays: pd.DataFrame) -> dict:
    """Per part and side: reliability, true sd, noise per play, carryover."""
    snaps_per_half = plays.groupby(["year", "offense"]).snaps.sum().median() / 2
    out = {}
    print(f"\n{'part':5s} {'side':8s} {'half r':>7s} {'season rel':>10s} {'true sd':>8s}"
          f" {'noise/play':>10s} {'carryover':>9s} {'r(t,t+1)':>9s}")
    for (part, side), h in halves.groupby(["part", "side"]):
        # Centered within season: a season's level is the league's, not a team's.
        a = h.a - h.groupby("year").a.transform("mean")
        b = h.b - h.groupby("year").b.transform("mean")
        r = float(np.corrcoef(a, b)[0, 1])
        true_var = float(np.mean(a * b))
        noise_var = float((np.mean(a * a) + np.mean(b * b)) / 2 - true_var)
        s = seasons[(seasons.part == part) & (seasons.side == side)]
        sa = s.a - s.groupby("year").a.transform("mean")
        sb = s.b - s.groupby("year").b.transform("mean")
        carry = float(np.mean(sa * sb)) / true_var
        out[part, side] = {"true_var": true_var, "per_play": noise_var * snaps_per_half}
        print(f"{part:5s} {side:8s} {r:7.3f} {2 * r / (1 + r):10.3f} {np.sqrt(true_var):8.4f}"
              f" {np.sqrt(noise_var * snaps_per_half):10.3f} {carry:9.3f}"
              f" {float(np.corrcoef(sa, sb)[0, 1]):9.3f}")
    return out


def margin_prior(games: pd.DataFrame, rng: np.random.Generator) -> float:
    """The margins' own noise-to-signal ratio, in games, measured the same way."""
    a_all, b_all = [], []
    for _, season in games.groupby("year"):
        ids = season.game_id.unique()
        for _ in range(2):
            half = set(rng.choice(ids, len(ids) // 2, replace=False))
            ra, na = rate_margins(season[season.game_id.isin(half)], 1.0)
            rb, nb = rate_margins(season[~season.game_id.isin(half)], 1.0)
            for t in ra:
                if na.get(t, 0) >= MIN_HALF and nb.get(t, 0) >= MIN_HALF and t in rb:
                    a_all.append(ra[t])
                    b_all.append(rb[t])
    a, b = np.array(a_all), np.array(b_all)
    true_var = float(np.mean((a - a.mean()) * (b - b.mean())))
    noise = float(np.var(a) + np.var(b)) / 2 - true_var
    sides = pd.concat([games[["year", "home_team"]].set_axis(["year", "team"], axis=1),
                       games[["year", "away_team"]].set_axis(["year", "team"], axis=1)])
    games_per_half = sides.groupby(["year", "team"]).size().median() / 2
    return noise * games_per_half / true_var


def features(games: pd.DataFrame, plays: pd.DataFrame, priors: dict, m_prior: float) -> pd.DataFrame:
    """Season-to-date fits before each week, as gaps for that week's games."""
    out = []
    for year, season in games.groupby("year"):
        season_plays = plays[plays.year == year]
        for week in sorted(season.week.unique()):
            if week < FIRST_WEEK:
                continue
            before = season_plays[season_plays.week < week]
            now = season[season.week == week]
            if before.empty:
                continue
            fits = {}
            for part in PARTS:
                o, d, n = fit(before, part, "snaps", priors[part])
                fits[part] = (o, d)
            seen = n
            r, _ = rate_margins(season[season.week < week], m_prior)
            for g in now.itertuples():
                h, a = g.home_team, g.away_team
                if seen.get(h, 0) < 3 or seen.get(a, 0) < 3 or h not in r or a not in r:
                    continue
                row = {"game_id": g.game_id, "year": year, "week": week, "fbs": g.fbs,
                       "home": g.home, "margin": g.margin, "win": float(g.margin > 0),
                       "tie": g.margin == 0, "prob": g.team1_win_prob, "rating": r[h] - r[a]}
                for part, (o, d) in fits.items():
                    row[part] = (o[h] - d[a]) - (o[a] - d[h])
                out.append(row)
    return pd.DataFrame(out)


def predict(df: pd.DataFrame) -> None:
    df = df[df.fbs].copy()
    eps = 1e-6
    df["prod"] = np.log((df.prob + eps) / (1 - df.prob + eps))
    train, test = df[df.year <= TRAIN_THROUGH], df[df.year > TRAIN_THROUGH]
    print(f"\nnext-week margins, week {FIRST_WEEK}+ : {len(train)} games through"
          f" {TRAIN_THROUGH}, {len(test)} after")
    specs = {
        "margins": ["home", "rating"],
        "margins + EPA": ["home", "rating", "epa"],
        "margins + efficiency": ["home", "rating", "eff"],
        "margins + eff + expl": ["home", "rating", "eff", "expl"],
        "eff + expl, no margins": ["home", "eff", "expl"],
        "published": ["prod"],
        "published + EPA": ["prod", "epa"],
        "published + eff + expl": ["prod", "eff", "expl"],
    }
    print(f"{'features':24s} {'test RMSE':>9s} {'test brier':>10s}   coefficients (points per unit)")
    for name, cols in specs.items():
        ols = LinearRegression().fit(train[cols], train.margin)
        rmse = float(np.sqrt(np.mean((ols.predict(test[cols]) - test.margin) ** 2)))
        decided = train[~train.tie]
        logit = LogisticRegression(C=1e6, max_iter=1000).fit(decided[cols], decided.win)
        brier = float(np.mean((logit.predict_proba(test[cols])[:, 1] - test.win) ** 2))
        # Standard errors for the margin fit, plain OLS.
        design = np.c_[np.ones(len(train)), train[cols]]
        resid = train.margin - ols.predict(train[cols])
        cov = np.linalg.inv(design.T @ design) * resid.var()
        coefs = "  ".join(f"{c} {b:+.2f} (t {b / np.sqrt(cov[i + 1, i + 1]):+.1f})"
                          for i, (c, b) in enumerate(zip(cols, ols.coef_)))
        print(f"{name:24s} {rmse:9.3f} {brier:10.5f}   {coefs}")


def main() -> None:
    league = sys.argv[1]
    rng = np.random.default_rng(0)
    games, plays = load(league)
    halves = reliability(plays, rng)
    seasons = carryover(plays)
    measured = summarize(halves, seasons, plays)
    # Each part's own prior, in plays: its noise per play over its true
    # variance, offense and defense averaged.
    priors = {part: np.mean([measured[part, s]["per_play"] / measured[part, s]["true_var"]
                             for s in ("offense", "defense")]) for part in PARTS}
    m_prior = margin_prior(games, rng)
    print("\nshrinkage, plays of prior weight:", {k: round(v) for k, v in priors.items()},
          f"margins {m_prior:.1f} games")
    predict(features(games, plays, priors, m_prior))


if __name__ == "__main__":
    main()
