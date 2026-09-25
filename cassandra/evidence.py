"""Test a proposed cause against the residuals, before building anything on it.

An idea for the model usually arrives as a story about a few teams -- the
coach left, the quarterback transferred in, the schedule front-loaded road
games -- and the way to find out whether it is a story or a signal is to
state it as a feature of *every* team-season (or every game) and ask the
residuals. This module is that ask, done the same way each time, so that
the answer for one idea can be compared with the answer for the last one.
It is what the offseason study did by hand with a directory of scratch
scripts (`coach_left_for_job` -2.7 +/- 0.7, a new quarterback -1.1 early),
made into one call.

The shape of the answer
-----------------------

For each feature and each window of a team's season -- its first
`EARLY_GAMES` games, the rest, and all of it -- `FeatureEffect` has:

- **`slope`**: points of own-side residual per unit of the feature. For a
  yes/no feature it is the difference in mean residual between the team-
  seasons with it and those without; negative means the model over-rates
  teams with it. With an intercept, so a window that runs high overall
  isn't credited to the feature.
- **`se`, `t`**: cluster-robust by team-season. A feature of a team-season
  is one draw per team-season however many games it played, and an
  ordinary per-game standard error would count a twelve-game season as
  twelve independent observations of it.
- **`perm_p`**: the share of within-season shuffles of the feature across
  team-seasons that produce a slope at least as large. The check on the
  standard error that doesn't depend on getting the error structure
  right -- a schedule's opponents, a conference's shared games.
- **`market_slope`**: the slope of `market - model` on the same feature,
  over the games a book priced. If the market moves with the feature the
  way the residual does, the market already prices it -- which is both
  confirmation that the effect is real and a measure of what the model is
  leaving on the table. Thin: the line history is a season or two.

Then `cross_validated_gain`: the joint fit of every feature, window by
window, fitted on odd seasons and applied to even ones and the reverse, as
a shift of each team's predicted margin. The MAE and brier change it makes
is the number to compare with a search's improvement -- a knob that shifts
a team's rating before the season can do about this well and usually a
little worse, since a rating shift fades as games arrive where this one is
applied flat within its window.

What it assumes
---------------

That the feature was knowable when the games were played. The residuals
can't tell a preseason fact from one read off the season it explains, and a
feature built from the season's own results -- "teams that went to a bowl"
-- will explain the residuals perfectly and predict nothing.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from .residuals import MARGIN_RESIDUAL, PREDICTED_MARGIN
from .team_seasons import EARLY_GAMES

#: Within-season shuffles behind `perm_p`.
DEFAULT_PERMUTATIONS = 500

_EPS = 1e-6


def windows(early_games: int = EARLY_GAMES) -> dict[str, tuple[int, int]]:
    """The slices of a team's season each effect is measured in, by game number."""
    return {
        f"games 1-{early_games}": (1, early_games),
        f"games {early_games + 1}+": (early_games + 1, 10**6),
        "all games": (1, 10**6),
    }


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Team-season features as numbers: booleans as 0/1, text one-hot by level.

    `features` has `team` and `year` and anything else. A text column
    becomes one 0/1 column per level, named `<column>=<level>`, each against
    every other row -- including the rows where the column is empty, which
    for a column like a coach's departure reason means "no departure". A
    missing number stays missing and leaves that feature's fit.
    """
    out = features[["team", "year"]].copy()
    for column in features.columns.drop(["team", "year"]):
        values = features[column]
        if values.dtype == bool or values.dtype == "boolean":
            out[column] = values.astype(float)
        elif pd.api.types.is_numeric_dtype(values):
            out[column] = values.astype(float)
        else:
            for level in sorted(values.dropna().unique()):
                out[f"{column}={level}"] = (values == level).astype(float)
    if out.duplicated(["team", "year"]).any():
        raise ValueError("features have more than one row for a team-season")
    return out


class FeatureEffect(NamedTuple):
    """One feature's effect in one window. See the module docstring for the columns.

    `nonzero` is how many team-seasons (games, for a game feature) have the
    feature at all. Under twenty or so, the cluster-robust `t` is not to be
    trusted -- a standard error estimated from three clusters is mostly
    those three clusters' luck -- and `perm_p` is the number to read.
    """

    feature: str
    window: str
    team_seasons: int
    games: int
    nonzero: int
    feature_mean: float
    feature_sd: float
    slope: float
    se: float
    t: float
    perm_p: float
    n_lined: int
    market_slope: float
    market_se: float


def _ols_clustered(
    x: np.ndarray, y: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(coefficients, cluster-robust standard errors) of y on [1, x].

    CR1: the sandwich with each cluster's score summed, scaled by
    G/(G-1) * (N-1)/(N-K) the way Stata's default is. The intercept is
    first in both returned arrays.
    """
    design = np.column_stack([np.ones(len(y)), x])
    bread = np.linalg.pinv(design.T @ design)
    beta = bread @ design.T @ y
    residual = y - design @ beta
    codes, uniques = pd.factorize(clusters)
    scores = np.zeros((len(uniques), design.shape[1]))
    np.add.at(scores, codes, design * residual[:, None])
    groups, n, k = len(uniques), len(y), design.shape[1]
    correction = (groups / max(groups - 1, 1)) * ((n - 1) / max(n - k, 1))
    covariance = correction * bread @ (scores.T @ scores) @ bread
    return beta, np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def _slope_from_sums(x: np.ndarray, n: np.ndarray, sums: np.ndarray) -> float:
    """The per-game OLS slope, from team-season totals, for a feature constant per team-season."""
    total = n.sum()
    x_bar = (n * x).sum() / total
    y_bar = sums.sum() / total
    denominator = (n * (x - x_bar) ** 2).sum()
    if denominator <= 0:
        return float("nan")
    return float(((x - x_bar) * (sums - n * y_bar)).sum() / denominator)


def _permutation_p(
    per_season: pd.DataFrame, feature: str, observed: float, draws: int, seed: int
) -> float:
    """Share of within-year shuffles of `feature` with a slope at least `observed`."""
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    x = per_season[feature].to_numpy()
    n = per_season["n"].to_numpy(dtype=float)
    sums = per_season["sum"].to_numpy()
    blocks = [
        np.flatnonzero(per_season["year"].to_numpy() == y)
        for y in per_season["year"].unique()
    ]
    shuffled = x.copy()
    hits = 0
    for _ in range(draws):
        for block in blocks:
            shuffled[block] = x[rng.permutation(block)]
        if abs(_slope_from_sums(shuffled, n, sums)) >= abs(observed) - _EPS:
            hits += 1
    return (hits + 1) / (draws + 1)


def feature_effects(
    games: pd.DataFrame,
    features: pd.DataFrame,
    early_games: int = EARLY_GAMES,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> list[FeatureEffect]:
    """Every feature's effect on the own-side residual, one feature at a time, per window.

    `games` is `team_seasons.team_games`, already restricted to the team-
    seasons the question is about (FBS since 2015, say); `features` is
    `prepare_features`'. Only team-seasons in both are used, and each
    feature only where it isn't missing.
    """
    joined = games.merge(features, on=["team", "year"], how="inner")
    names = [c for c in features.columns if c not in ("team", "year")]
    effects = []
    for window, (first, last) in windows(early_games).items():
        rows = joined[joined["game_number"].between(first, last)]
        for name in names:
            sample = rows[rows[name].notna()]
            if sample.empty:
                continue
            x = sample[name].to_numpy(dtype=float)
            y = sample["residual"].to_numpy(dtype=float)
            cluster = sample["team"].astype(str) + "\x00" + sample["year"].astype(str)
            if np.ptp(x) == 0:
                continue
            beta, se = _ols_clustered(x, y, cluster.to_numpy())
            per_season = (
                sample.groupby(["team", "year"])
                .agg(
                    n=("residual", "size"),
                    sum=("residual", "sum"),
                    **{name: (name, "first")},
                )
                .reset_index()
            )
            lined = sample[sample["market"].notna()]
            market_slope, market_se = float("nan"), float("nan")
            if len(lined) > 2 and np.ptp(lined[name]) > 0:
                gap = (lined["market"] - lined["predicted"]).to_numpy(dtype=float)
                lined_clusters = (
                    lined["team"].astype(str) + "\x00" + lined["year"].astype(str)
                ).to_numpy()
                m_beta, m_se = _ols_clustered(
                    lined[name].to_numpy(dtype=float), gap, lined_clusters
                )
                market_slope, market_se = float(m_beta[1]), float(m_se[1])
            effects.append(
                FeatureEffect(
                    feature=name,
                    window=window,
                    team_seasons=len(per_season),
                    games=len(sample),
                    nonzero=int((per_season[name] != 0).sum()),
                    feature_mean=float(per_season[name].mean()),
                    feature_sd=float(per_season[name].std()),
                    slope=float(beta[1]),
                    se=float(se[1]),
                    t=float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
                    perm_p=_permutation_p(
                        per_season, name, float(beta[1]), permutations, seed
                    ),
                    n_lined=len(lined),
                    market_slope=market_slope,
                    market_se=market_se,
                )
            )
    return effects


class JointFit(NamedTuple):
    """Every feature at once, in one window: coefficients and their errors."""

    window: str
    games: int
    team_seasons: int
    coefficients: pd.DataFrame  # feature, slope, se, t


def joint_fits(
    games: pd.DataFrame, features: pd.DataFrame, early_games: int = EARLY_GAMES
) -> list[JointFit]:
    """The multiple regression of the residual on every feature, per window.

    The univariate slopes answer "is this feature alone associated with the
    error"; this answers "does it still matter given the others", which is
    the question when two features travel together -- a new quarterback and
    the change in quarterback quality, say. Rows missing any feature drop.
    """
    names = [c for c in features.columns if c not in ("team", "year")]
    joined = games.merge(features, on=["team", "year"], how="inner").dropna(
        subset=names
    )
    fits = []
    for window, (first, last) in windows(early_games).items():
        sample = joined[joined["game_number"].between(first, last)]
        usable = (
            [n for n in names if np.ptp(sample[n].to_numpy()) > 0]
            if len(sample)
            else []
        )
        if not usable:
            continue
        cluster = (
            sample["team"].astype(str) + "\x00" + sample["year"].astype(str)
        ).to_numpy()
        beta, se = _ols_clustered(
            sample[usable].to_numpy(dtype=float),
            sample["residual"].to_numpy(dtype=float),
            cluster,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            t = beta[1:] / se[1:]
        fits.append(
            JointFit(
                window=window,
                games=len(sample),
                team_seasons=len(set(cluster)),
                coefficients=pd.DataFrame(
                    {"feature": usable, "slope": beta[1:], "se": se[1:], "t": t}
                ),
            )
        )
    return fits


class Gain(NamedTuple):
    """What shifting predicted margins by a fitted feature effect did out of sample.

    Negative is better for both. `games` is how many games the evaluation
    covered: those with at least one side in the feature table and its
    filter, which is the set a search restricted the same way would score.
    `touched` is how many of them the shift moved at all.
    """

    games: int
    touched: int
    mae_before: float
    mae_change: float
    brier_before: float
    brier_change: float


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def margin_scale(scored: pd.DataFrame) -> float:
    """Points per logit of the margin fit the residuals were taken through.

    `add_residuals`' MAE logistic is `predicted = scale * logit(p)`, so the
    scale is recoverable from the frame; a shift of `d` points is a shift of
    `d / scale` in logit.
    """
    logits = _logit(scored["team1_win_prob"].to_numpy(dtype=float))
    predicted = scored[PREDICTED_MARGIN].to_numpy(dtype=float)
    return float(logits @ predicted / (logits @ logits))


def cross_validated_gain(
    scored: pd.DataFrame,
    games: pd.DataFrame,
    features: pd.DataFrame,
    early_games: int = EARLY_GAMES,
) -> Gain:
    """Fit every feature jointly on odd seasons, shift even ones by it, and the reverse.

    Per window (`games 1-N` and `games N+1+` separately, so an effect that
    fades is applied where it was measured), as a shift of each side's own
    predicted margin: `slope . features` for the home team minus the same
    for the away team, zero for a side with no row in `features`. Scored on
    every game `games` has a side of, in points of MAE and in brier through
    `margin_scale`.
    """
    names = [c for c in features.columns if c not in ("team", "year")]
    filled = features.fillna({n: 0.0 for n in names})
    joined = games.merge(filled, on=["team", "year"], how="left")
    early_window = (1, early_games)
    late_window = (early_games + 1, 10**6)
    shift = np.zeros(len(joined))
    years = joined["year"].to_numpy()
    for parity in (0, 1):
        train = joined[(years % 2 == parity) & joined[names[0]].notna()]
        test = (years % 2 != parity) & joined[names[0]].notna().to_numpy()
        for first, last in (early_window, late_window):
            fit_rows = train[train["game_number"].between(first, last)]
            usable = [
                n for n in names if len(fit_rows) and np.ptp(fit_rows[n].to_numpy()) > 0
            ]
            if not usable:
                continue
            cluster = (
                fit_rows["team"].astype(str) + "\x00" + fit_rows["year"].astype(str)
            ).to_numpy()
            beta, _ = _ols_clustered(
                fit_rows[usable].to_numpy(dtype=float),
                fit_rows["residual"].to_numpy(dtype=float),
                cluster,
            )
            in_window = test & joined["game_number"].between(first, last).to_numpy()
            shift[in_window] = (
                joined.loc[in_window, usable].to_numpy(dtype=float) @ beta[1:]
            )
    # Back to one number per game: the home side's shift minus the away side's.
    signed = np.where(joined["home_side"].to_numpy(dtype=bool), shift, -shift)
    per_game = pd.Series(signed).groupby(joined["game_id"].to_numpy()).sum()
    covered = scored[scored["game_id"].astype(str).isin(set(games["game_id"]))]
    home_shift = per_game.reindex(covered["game_id"].astype(str)).fillna(0.0).to_numpy()

    mov = (covered["home_score"] - covered["away_score"]).to_numpy(dtype=float)
    predicted = covered[PREDICTED_MARGIN].to_numpy(dtype=float)
    p = covered["team1_win_prob"].to_numpy(dtype=float)
    won = (mov > 0).astype(float)
    shifted_p = 1 / (1 + np.exp(-(_logit(p) + home_shift / margin_scale(scored))))
    mae_before = float(np.abs(mov - predicted).mean())
    brier_before = float(((p - won) ** 2).mean())
    return Gain(
        games=len(covered),
        touched=int((np.abs(home_shift) > _EPS).sum()),
        mae_before=mae_before,
        mae_change=float(np.abs(mov - predicted - home_shift).mean()) - mae_before,
        brier_before=brier_before,
        brier_change=float(((shifted_p - won) ** 2).mean()) - brier_before,
    )


def game_feature_effects(
    scored: pd.DataFrame,
    features: pd.DataFrame,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> list[FeatureEffect]:
    """The same test for a feature of a *game*, read from the home side.

    `features` has `game_id` and numeric columns, each signed so that
    positive favours the home team -- a rest advantage in days, the
    visitor's travel in km. The residual is the home side's; the standard
    error is per game (a game is its own cluster), and the permutation
    shuffles the feature across games within a season.
    """
    joined = scored.assign(game_id=scored["game_id"].astype(str)).merge(
        features.assign(game_id=features["game_id"].astype(str)), on="game_id"
    )
    rng = np.random.default_rng(seed)
    effects = []
    for name in features.columns.drop("game_id"):
        sample = joined[joined[name].notna()]
        x = sample[name].to_numpy(dtype=float)
        if len(sample) < 3 or np.ptp(x) == 0:
            continue
        y = sample[MARGIN_RESIDUAL].to_numpy(dtype=float)
        beta, se = _ols_clustered(x, y, sample["game_id"].to_numpy())
        years = sample["year"].to_numpy()
        blocks = [np.flatnonzero(years == year) for year in np.unique(years)]
        shuffled, hits = x.copy(), 0
        for _ in range(permutations):
            for block in blocks:
                shuffled[block] = x[rng.permutation(block)]
            if abs(np.polyfit(shuffled, y, 1)[0]) >= abs(beta[1]) - _EPS:
                hits += 1
        market = -pd.to_numeric(sample["spread"], errors="coerce").to_numpy()
        lined = ~np.isnan(market)
        market_slope, market_se = float("nan"), float("nan")
        if lined.sum() > 2 and np.ptp(x[lined]) > 0:
            gap = market[lined] - sample[PREDICTED_MARGIN].to_numpy()[lined]
            m_beta, m_se = _ols_clustered(
                x[lined], gap, sample["game_id"].to_numpy()[lined]
            )
            market_slope, market_se = float(m_beta[1]), float(m_se[1])
        effects.append(
            FeatureEffect(
                feature=name,
                window="all games",
                team_seasons=0,
                games=len(sample),
                nonzero=int((x != 0).sum()),
                feature_mean=float(x.mean()),
                feature_sd=float(x.std()),
                slope=float(beta[1]),
                se=float(se[1]),
                t=float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
                perm_p=(hits + 1) / (permutations + 1),
                n_lined=int(lined.sum()),
                market_slope=market_slope,
                market_se=market_se,
            )
        )
    return effects


def buckets(
    values: pd.Series, residual: pd.Series, edges: Sequence[float] | None = None
) -> pd.DataFrame:
    """Mean residual by bucket of a numeric feature: quintiles unless `edges` are given.

    The check on a slope's shape. A slope is one number for a relationship
    that may live entirely in one tail -- the 35-point blowouts in the
    momentum finding -- and the buckets show where.
    """
    frame = pd.DataFrame({"x": values, "residual": residual}).dropna()
    if edges is None:
        labels = pd.qcut(frame["x"], 5, duplicates="drop")
    else:
        labels = pd.cut(frame["x"], [-np.inf, *edges, np.inf])
    grouped = frame.groupby(labels, observed=True)["residual"]
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "mean": grouped.mean(),
            "se": grouped.std() / np.sqrt(grouped.size()),
        }
    ).reset_index(names="bucket")
