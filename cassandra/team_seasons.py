"""Which team-seasons the model keeps getting wrong, and how much of that is real.

`cassandra.residuals` asks whether the error has structure along an axis --
a week, a kickoff slot, a tier matchup. This asks it of the unit people
actually talk about: a team's season. "The model has been under Indiana all
year" is a claim about one team-season's residuals, and it is the claim a new
idea usually starts from -- find the teams the model keeps missing, then ask
what they have in common.

The trap is that a football season is short and a game is noisy. With a
per-game residual sd around 16 points on ncaafb, twelve games leave a
team-season's mean residual with a standard error of about 4.6 -- and the
2026-09-24 offseason study found the whole spread of team-season means was
4.7. Almost all of what makes a team-season look "consistently" missed is
the schedule's luck. So every number here comes with what noise alone would
have produced, and the ranking is on an estimate that has already had the
noise taken out.

Three estimates of a team-season's error, each answering its own question
---------------------------------------------------------------------------

All in points of margin, signed from the team's side: positive means the
team beat what the model expected of it, i.e. the model *under-rated* it.

- **`raw`** -- the mean of the team's own-side residuals. What the season
  looked like. `z` is it over its standard error, `raw / (sigma / sqrt(n))`.
- **`shrunk`** -- the persistent part, estimated. A ridge fit per season of
  every game's residual on `effect[home] - effect[away]`, with the ridge
  penalty set by how big persistent team-season effects actually are across
  the league's history (`tau`, fit by marginal likelihood). Two things
  happen at once: an opponent's own error is netted out, so beating a
  team the model over-rated isn't credited as being under-rated yourself;
  and a season's mean is pulled toward zero by exactly as much as its
  noise deserves. This is the number to rank on.
- **`p_sign`** -- the posterior probability that the true effect has the
  sign `shrunk` has. Near 0.5 is a coin; above 0.9 is a team-season worth
  reading about.

And the context that says what kind of miss it is:

- **`early` / `late`** -- the raw mean in the team's first `EARLY_GAMES`
  games and after. An error that lives early is something the model could
  have known before the season (a quarterback, a coach, a roster); one
  that persists late is the update not keeping up.
- **`n_lined`, `model_vs_market`, `market_residual`** -- over the games a
  book priced: how much more the model liked this team than the closing
  line did, and how much the team beat the line. A team the model under-
  rates and the market *didn't* (`market_residual` near zero) is a team the
  market knew something about. A team that beat both is a genuine surprise,
  and nothing pre-game would have found it.

What to read first
------------------

`SeasonSummary`: `tau` against the noise per team-season. If the persistent
spread is 1.2 points and a season's noise is 4.6, a team-season is on
average only about 6% signal and the top of any ranked list is mostly the
luckiest schedules. `null_counts` says the same thing as a count: how many
team-seasons past |z| of 2 and 3 there are, against how many random sign
flips of the same games produce.

What it won't do
----------------

Name a cause. A ranked list of the team-seasons the model missed most is a
list of places to look, and the looking -- the coach who left, the
quarterback who transferred in -- is `evidence.py`'s job, which tests a
proposed cause against every team-season at once rather than the dozen that
suggested it.
"""

import math
from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from .residuals import (
    MARGIN_RESIDUAL,
    MARKET_MARGIN,
    PREDICTED_MARGIN,
    Tier,
    conference_label,
    division_label,
)

#: A team's first this-many games of a season are its `early` games. Four,
#: because that is where the 2026-09 residual sweep and the offseason study
#: both found the preseason's error concentrated.
EARLY_GAMES = 4

#: |z| cut-offs `null_counts` reports at.
Z_THRESHOLDS = (2.0, 3.0)

#: Sign-flip draws behind a null count. The count's own noise is what
#: matters, and a couple of hundred draws pin its mean to a few percent.
DEFAULT_NULL_DRAWS = 200

#: The persistent spreads `fit_tau` considers, in points.
TAU_GRID = np.arange(0.0, 10.001, 0.05)


def team_games(scored: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, one per team, everything signed to that team's side.

    `scored` has been through `residuals.add_residuals`. Columns:
    `team`, `opponent`, `year`, `week_number`, `date`, `game_id`, `site`
    (home / away / neutral), `home_side` (whether this row is the frame's
    home team, which a neutral site still has one of), `game_number` (the
    team's nth game of the
    season, by kickoff), `predicted` and `actual` margins, `residual`
    (actual minus predicted), `market` (the margin the closing line implied,
    NaN where there was none), `market_residual` (actual minus market) and
    `model_minus_market`.
    """
    n = len(scored)
    neutral = scored["neutral_site"].to_numpy(dtype=bool)
    predicted = scored[PREDICTED_MARGIN].to_numpy(dtype=float)
    actual = (scored["home_score"] - scored["away_score"]).to_numpy(dtype=float)
    residual = scored[MARGIN_RESIDUAL].to_numpy(dtype=float)
    market = scored[MARKET_MARGIN].to_numpy(dtype=float)
    dates = pd.to_datetime(scored["date"], utc=True).to_numpy()

    def side(team: str, opponent: str, sign: float, label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "team": scored[team].to_numpy(),
                "opponent": scored[opponent].to_numpy(),
                "year": scored["year"].to_numpy(),
                "week_number": scored["week_number"].to_numpy(),
                "date": dates,
                "game_id": scored["game_id"].astype(str).to_numpy(),
                "site": np.where(neutral, "neutral", label),
                "home_side": sign > 0,
                "position": np.arange(n),
                "predicted": sign * predicted,
                "actual": sign * actual,
                "residual": sign * residual,
                "market": sign * market,
            }
        )

    long = pd.concat(
        [
            side("home_team", "away_team", 1.0, "home"),
            side("away_team", "home_team", -1.0, "away"),
        ],
        ignore_index=True,
    ).sort_values(["date", "position"], kind="stable")
    long["game_number"] = long.groupby(["team", "year"], sort=False).cumcount() + 1
    long["market_residual"] = long["actual"] - long["market"]
    long["model_minus_market"] = long["predicted"] - long["market"]
    return long.drop(columns="position").reset_index(drop=True)


class _Season(NamedTuple):
    """One season's games as a ridge problem, factored once.

    The design is games x labels, +1 for the home side's label and -1 for
    the away side's; games whose two sides share a label carry nothing about
    either and are left out. Only what the fit needs of it is kept -- the
    eigendecomposition of `X'X` (`values`, `vectors`), `X'r` projected onto
    those vectors, and `r'r` -- which is what lets `tau` be searched over a
    grid without a solve per step, and keeps ncaafb's 25 seasons of
    ~700 teams x ~3,000 games out of memory.
    """

    year: int
    labels: list[Hashable]
    games: int
    rss: float
    projected: np.ndarray
    values: np.ndarray
    vectors: np.ndarray


def _seasons(scored: pd.DataFrame, home: np.ndarray, away: np.ndarray) -> list[_Season]:
    residual = scored[MARGIN_RESIDUAL].to_numpy(dtype=float)
    years = scored["year"].to_numpy()
    seasons = []
    for year in np.unique(years):
        rows = np.flatnonzero((years == year) & (home != away))
        if rows.size == 0:
            continue
        labels, codes = np.unique(
            np.concatenate([home[rows], away[rows]]), return_inverse=True
        )
        h, a, r = codes[: rows.size], codes[rows.size :], residual[rows]
        k = labels.size
        gram = np.zeros((k, k))
        np.add.at(gram, (h, h), 1.0)
        np.add.at(gram, (a, a), 1.0)
        np.add.at(gram, (h, a), -1.0)
        np.add.at(gram, (a, h), -1.0)
        xtr = np.bincount(h, weights=r, minlength=k) - np.bincount(
            a, weights=r, minlength=k
        )
        values, vectors = np.linalg.eigh(gram)
        seasons.append(
            _Season(
                year=int(year),
                labels=labels.tolist(),
                games=int(rows.size),
                rss=float(r @ r),
                projected=vectors.T @ xtr,
                values=np.clip(values, 0.0, None),
                vectors=vectors,
            )
        )
    return seasons


def _log_likelihood(seasons: Sequence[_Season], tau: float, sigma2: float) -> float:
    """Marginal log likelihood of every season's residuals, up to a constant.

    residual ~ N(0, sigma2 I + tau^2 X X'), evaluated in label space through
    the eigendecomposition of X'X (Woodbury and the matrix determinant lemma)
    so a season of ncaafb costs a few hundred operations, not a solve on a
    few thousand games.
    """
    total = 0.0
    for season in seasons:
        rss = season.rss
        if tau > 0:
            ratio = sigma2 / tau**2
            rss -= float(np.sum(season.projected**2 / (season.values + ratio)))
            total -= float(np.sum(np.log1p(season.values * tau**2 / sigma2)))
        total -= rss / sigma2 + season.games * math.log(sigma2)
    return total / 2


def fit_tau(seasons: Sequence[_Season]) -> tuple[float, float]:
    """(tau, sigma): the persistent spread of season effects, and the game noise.

    tau by marginal likelihood over `TAU_GRID`. sigma is the residual's own
    sd with the two sides' persistent parts taken out, since a game's
    residual carries both teams' effects -- one refinement pass is enough,
    the correction being a percent or so of sigma.
    """
    total_var = sum(s.rss for s in seasons) / sum(s.games for s in seasons)
    sigma2, tau = total_var, 0.0
    for _ in range(2):
        scores = [_log_likelihood(seasons, t, sigma2) for t in TAU_GRID]
        tau = float(TAU_GRID[int(np.argmax(scores))])
        sigma2 = max(total_var - 2 * tau**2, total_var / 4)
    return tau, math.sqrt(sigma2)


class SeasonEffects(NamedTuple):
    """Every (year, label)'s shrunk effect, and the spreads it was shrunk with."""

    effects: pd.DataFrame  # year, label, shrunk, posterior_sd, p_sign
    tau: float
    sigma: float


def season_effects(
    scored: pd.DataFrame,
    home: np.ndarray,
    away: np.ndarray,
    tau: float | None = None,
) -> SeasonEffects:
    """The ridge estimate of a per-season effect for each label, opponents netted out.

    `home` and `away` label each game's two sides -- team names for a team-
    season, conference names for a conference-season, anything hashable.
    `tau` fixes the prior spread instead of fitting it.

    Effects are relative to the season's average label. A game's residual
    only says how the two sides compare, so adding a constant to every
    effect changes nothing the games can see, and the ridge settles that by
    centring each season (each connected part of its schedule) on zero. A
    team 12 points worse than rated in a 20-team season comes out at about
    -11.4, with +0.6 spread across everyone else.
    """
    seasons = _seasons(
        scored, np.asarray(home, dtype=object), np.asarray(away, dtype=object)
    )
    if not seasons:
        empty = pd.DataFrame(
            columns=["year", "label", "shrunk", "posterior_sd", "p_sign"]
        )
        return SeasonEffects(empty, 0.0, float("nan"))
    fitted_tau, sigma = fit_tau(seasons)
    tau = fitted_tau if tau is None else tau
    rows = []
    for season in seasons:
        if tau > 0:
            inverse = 1.0 / (season.values + sigma**2 / tau**2)
            estimate = season.vectors @ (inverse * season.projected)
            variance = sigma**2 * (season.vectors**2 @ inverse)
        else:
            estimate = np.zeros(len(season.labels))
            variance = np.zeros(len(season.labels))
        sd = np.sqrt(variance)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_sign = np.where(sd > 0, _normal_cdf(np.abs(estimate) / sd), 0.5)
        rows.append(
            pd.DataFrame(
                {
                    "year": season.year,
                    "label": season.labels,
                    "shrunk": estimate,
                    "posterior_sd": sd,
                    "p_sign": p_sign,
                }
            )
        )
    return SeasonEffects(pd.concat(rows, ignore_index=True), tau, sigma)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


class SeasonSummary(NamedTuple):
    """How much of a team-season's error is the team's, over a set of them.

    `observed_sd` is the game-weighted spread of team-season raw means;
    `noise_sd` is what `sigma / sqrt(n)` alone would give on the same
    seasons, and `excess_sd` is what's left of the first once the second is
    taken out in quadrature -- the persistent spread of *this* set, measured
    directly. `tau` is the league-wide persistent spread the ridge fit
    shrank with, and `reliability`, `tau^2 / (tau^2 + noise_sd^2)`, is the
    share of a typical team-season in the set that is signal. The two
    correlations are the tests the offseason study ran by hand: a
    team-season's early error against its late one (does a miss persist
    within a season?) and a team's error one season against the next (does
    it persist across them?).
    """

    team_seasons: int
    sigma: float
    tau: float
    observed_sd: float
    noise_sd: float
    excess_sd: float
    reliability: float
    early_late_corr: float
    year_to_year_corr: float


class TeamSeasons(NamedTuple):
    """`team_season_table`'s rows, and the league-wide spreads behind `shrunk`."""

    table: pd.DataFrame
    tau: float
    sigma: float


def team_season_table(
    scored: pd.DataFrame,
    tiers: Callable[[str, int], Tier] | None = None,
    early_games: int = EARLY_GAMES,
) -> TeamSeasons:
    """One row per team-season: the three estimates and the context.

    `tiers` is `residuals.team_tiers` for a classified league, and adds
    `division` and `conference` columns; None leaves them out. Sorted by
    `shrunk`, most under-rated first. The ridge behind `shrunk` is fit on
    every game in `scored`, so filter the table, not the frame, to look at
    a subset: a team's opponents' errors are netted out wherever they play.
    """
    games = team_games(scored)
    effects = season_effects(
        scored, scored["home_team"].to_numpy(), scored["away_team"].to_numpy()
    )
    sigma = effects.sigma

    grouped = games.assign(
        beat=games["residual"] > 0, lined=games["market"].notna()
    ).groupby(["team", "year"], sort=False)
    table = grouped.agg(
        n=("residual", "size"),
        raw=("residual", "mean"),
        beat=("beat", "mean"),
        n_lined=("lined", "sum"),
        model_vs_market=("model_minus_market", "mean"),
        market_residual=("market_residual", "mean"),
    )
    early = games["game_number"] <= early_games
    table["early"] = games[early].groupby(["team", "year"])["residual"].mean()
    table["late"] = games[~early].groupby(["team", "year"])["residual"].mean()
    table = table.reset_index()
    table["z"] = table["raw"] / (sigma / np.sqrt(table["n"]))

    shrunk = effects.effects.rename(columns={"label": "team"})
    table = label_tiers(table.merge(shrunk, on=["team", "year"], how="left"), tiers)
    table = table.sort_values("shrunk", ascending=False, kind="stable")
    return TeamSeasons(table.reset_index(drop=True), effects.tau, sigma)


def group_season_table(
    scored: pd.DataFrame,
    home: Sequence[str] | pd.Series | np.ndarray,
    away: Sequence[str] | pd.Series | np.ndarray,
) -> TeamSeasons:
    """The same estimates for groups of teams -- conferences, divisions -- per season.

    `home` and `away` label each game's sides with their group. Only games
    between two groups say anything about either one's level, since inside
    a group both sides' shares of a shared error cancel; so `n` counts those
    games alone, and a conference whose teams the model over-rates together
    shows up only in what its non-conference schedule did. That is the
    shape the 2026 cross-tier miss had, and a per-team table spreads it
    across a hundred team-seasons too thin to see.
    """
    home_labels = np.asarray(home, dtype=object)
    away_labels = np.asarray(away, dtype=object)
    effects = season_effects(scored, home_labels, away_labels)
    cross = home_labels != away_labels
    games = team_games(
        scored.assign(home_team=home_labels, away_team=away_labels).loc[cross]
    )
    table = (
        games.assign(lined=games["market"].notna())
        .groupby(["team", "year"], sort=False)
        .agg(
            n=("residual", "size"),
            raw=("residual", "mean"),
            n_lined=("lined", "sum"),
            model_vs_market=("model_minus_market", "mean"),
            market_residual=("market_residual", "mean"),
        )
        .reset_index()
    )
    table["z"] = table["raw"] / (effects.sigma / np.sqrt(table["n"]))
    table = table.merge(
        effects.effects.rename(columns={"label": "team"}), on=["team", "year"]
    ).rename(columns={"team": "group"})
    table = table.sort_values("shrunk", ascending=False, kind="stable")
    return TeamSeasons(table.reset_index(drop=True), effects.tau, effects.sigma)


def summarize(table: pd.DataFrame, tau: float, sigma: float) -> SeasonSummary:
    """The `SeasonSummary` of a (possibly filtered) `team_season_table`."""
    weights = table["n"]
    noise_sd = float(np.sqrt(np.average(sigma**2 / weights, weights=weights)))
    centered = table["raw"] - np.average(table["raw"], weights=weights)
    observed_sd = float(np.sqrt(np.average(centered**2, weights=weights)))
    both = table.dropna(subset=["early", "late"])
    return SeasonSummary(
        team_seasons=len(table),
        sigma=sigma,
        tau=tau,
        observed_sd=observed_sd,
        noise_sd=noise_sd,
        excess_sd=math.sqrt(max(observed_sd**2 - noise_sd**2, 0.0)),
        reliability=tau**2 / (tau**2 + noise_sd**2) if noise_sd else float("nan"),
        early_late_corr=_corr(both["early"], both["late"]),
        year_to_year_corr=_year_to_year(table),
    )


def _corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _year_to_year(table: pd.DataFrame) -> float:
    following = table[["team", "year", "raw"]].assign(year=table["year"] - 1)
    pairs = table[["team", "year", "raw"]].merge(
        following, on=["team", "year"], suffixes=("", "_next")
    )
    return _corr(pairs["raw"], pairs["raw_next"])


def null_counts(
    scored: pd.DataFrame,
    table: pd.DataFrame,
    sigma: float,
    draws: int = DEFAULT_NULL_DRAWS,
    thresholds: Sequence[float] = Z_THRESHOLDS,
    seed: int = 0,
) -> pd.DataFrame:
    """How many team-seasons pass each |z|, observed and under random sign flips.

    The null flips each *game's* residual, both sides together, so a flip
    keeps the schedule -- who played whom, how often -- and destroys only
    the question of which way each game went. Under no persistent team
    effect at all, each residual is as likely to have gone the other way,
    so this is the count of "consistently missed" team-seasons a model with
    nothing to fix would still show. `table` is `team_season_table`'s, and
    restricts which team-seasons are counted (filter it first to count FBS
    only).
    """
    years = scored["year"].astype(str).to_numpy()
    home = scored["home_team"].astype(str).to_numpy() + "\x00" + years
    away = scored["away_team"].astype(str).to_numpy() + "\x00" + years
    codes, uniques = pd.factorize(np.concatenate([home, away]))
    home_codes, away_codes = codes[: len(scored)], codes[len(scored) :]
    wanted = pd.Index(uniques).isin(
        table["team"].astype(str) + "\x00" + table["year"].astype(str)
    )
    residual = scored[MARGIN_RESIDUAL].to_numpy(dtype=float)
    k = len(uniques)
    sizes = np.bincount(codes, minlength=k).astype(float)
    scale = sigma / np.sqrt(sizes)

    def z_of(flips: np.ndarray) -> np.ndarray:
        # A flip is of the game, so both of its sides flip with it: the home
        # side's residual is `r`, the away side's `-r`, both times `flips`.
        flipped = residual * flips
        sums = np.bincount(home_codes, weights=flipped, minlength=k) - np.bincount(
            away_codes, weights=flipped, minlength=k
        )
        return np.abs(sums / sizes / scale)[wanted]

    observed = z_of(np.ones(len(scored)))
    rng = np.random.default_rng(seed)
    null = np.array(
        [z_of(rng.choice((-1.0, 1.0), size=len(scored))) for _ in range(draws)]
    )
    rows = []
    for threshold in thresholds:
        counts = (null >= threshold).sum(axis=1)
        rows.append(
            {
                "threshold": threshold,
                "observed": int((observed >= threshold).sum()),
                "null_mean": float(counts.mean()),
                "null_p95": float(np.quantile(counts, 0.95)),
                "team_seasons": int(wanted.sum()),
            }
        )
    return pd.DataFrame(rows)


#: Short names a division can be asked for by, for the labels
#: call-it-what-you-want files ncaafb's lower tiers under. Anything else is
#: matched against the labels themselves, ignoring case.
DIVISION_ALIASES = {"d2": "NCAA Division II", "d3": "NCAA Division III"}


def label_tiers(
    frame: pd.DataFrame, tiers: Callable[[str, int], Tier] | None
) -> pd.DataFrame:
    """`frame` (with `team` and `year`) plus its `division` and `conference`.

    The conference is the name alone -- "Big Ten", not "FBS / Big Ten" --
    since the division is its own column. Unchanged when `tiers` is None.
    """
    if tiers is None:
        return frame
    found = [tiers(t, int(y)) for t, y in zip(frame["team"], frame["year"])]
    return frame.assign(
        division=[division_label(t) for t in found],
        conference=[conference_label(t).split(" / ", 1)[-1] for t in found],
    )


def select(
    frame: pd.DataFrame,
    division: str | None = None,
    since: int | None = None,
    until: int | None = None,
) -> pd.DataFrame:
    """The rows of a team-season or team-game frame in a division and a span of seasons.

    Raises `ValueError` naming the divisions on file when `division` isn't
    one of them, since the likeliest reason is a label spelled differently.
    """
    if since is not None:
        frame = frame[frame["year"] >= since]
    if until is not None:
        frame = frame[frame["year"] <= until]
    if division is not None:
        if "division" not in frame.columns:
            raise ValueError("no divisions on file for this league")
        wanted = DIVISION_ALIASES.get(division.lower(), division)
        labels = sorted(set(frame["division"]))
        match = [label for label in labels if label.lower() == wanted.lower()]
        if not match:
            raise ValueError(f"division {wanted!r} is none of {labels}")
        frame = frame[frame["division"] == match[0]]
    return frame


def game_log(games: pd.DataFrame, team: str, year: int | None = None) -> pd.DataFrame:
    """One team's games, in order, with what the model and the market said."""
    rows = games[games["team"] == team]
    if year is not None:
        rows = rows[rows["year"] == year]
    return rows[
        [
            "year",
            "game_number",
            "week_number",
            "date",
            "site",
            "opponent",
            "predicted",
            "market",
            "actual",
            "residual",
            "market_residual",
        ]
    ].reset_index(drop=True)
