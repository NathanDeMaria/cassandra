"""Where a model is wrong, so a signal can be chosen instead of guessed.

`model_eval` scores a model with one number per league. That is the right
shape for picking between models and the wrong shape for picking what to
build next: a margin MAE of 13.04 says the model is 13 points off on an
average NCAAFB game and says nothing at all about *which* games. Every
question that starts "should I add..." -- offense and defense apart, a home
advantage per team, recruiting, anything -- is a question about a subset, and
the aggregate has already averaged the subsets together.

So this splits the predictions the model already makes into slices and asks
one question of each: **does the model's error have structure along this
axis?** A signal is worth building when the answer is yes and the structure
is large enough to matter; it is not worth building when the residuals are
flat along it, however plausible the story. That ordering -- measure the
residual first, build the feature second -- is what `cassandra.predictor.
control` spent a year of negative results establishing, and this is the
cheap version of it: no new predictor, no new sweep, no search. One replay
of a model you already have.

The residual, and why it's the margin one
-----------------------------------------

`margin_residual` is the actual margin minus the margin the model's win
probability implies, through the same `MaeLogisticProbToMarginFitter` the
`margin_mae` objective scores by. Positive means the home team beat what the
model expected.

Fit **globally, then sliced**. That ordering is the whole design. A fit per
slice would absorb exactly the bias being looked for -- a week-1 fit that
learns "week 1 runs 2 points high" reports a week-1 bias of zero -- and the
question is whether one global model is systematically wrong somewhere, not
whether a model refit on each slice can be made right there. The global fit
is what ships; its residual is the honest one.

Margin rather than brier because brier is not comparable across slices. A
slice of lopsided games scores better than a slice of coin flips under a
perfect model, so a per-slice brier table ranks slices by how certain their
games were and tells you nothing about the model. Margin bias has no such
problem: under a correct model it is zero in every slice, whatever the games
were. `win_prob_bias` is reported next to it for the same reason it stays a
*bias* rather than a score -- mean(p) - mean(outcome) is zero everywhere
under a calibrated model too.

Three numbers per slice, and what each one is for
-------------------------------------------------

- **`margin_bias`** -- mean residual. The model is systematically wrong by
  this many points here. This is the number a feature could fix, and its
  sign says which way.
- **`margin_mae`** -- mean absolute residual. How noisy the slice is. Mostly
  a denominator: a bias of 0.5 points against an MAE of 13 is worth roughly
  nothing, and the same bias in a slice at MAE 8 is worth attention.
- **`market_gap`** -- the model's MAE minus the market's, over the games in
  the slice a book put a line on. Positive means the market beat the model
  here, by that many points.

`market_gap` is the most informative column in the table and deserves its
own paragraph. The line has already priced everything this module could ask
about -- injuries, a quarterback change, weather, who transferred in, what a
recruiting class did to a roster -- so the shape of the model's gap to the
market across an axis is a direct read on what the model is *missing*, not
merely on what it is noisy about. `margin_bias` finds a lever the model
could pull; `market_gap` says whether there is anything left to win by
pulling it. An axis where the bias is flat and the gap is flat has nothing on
it. An axis where the gap is wide early and closes later has something the
model could learn before the season starts, which is the exact shape the
recruiting-and-transfers question is asking about.

Its limits are the odds database's: the leagues and seasons a book was
quoted for, which is a minority of an NCAAFB history that starts in 1980.
`n_lined` is on every row so a gap over eleven games can be discounted on
sight, and it is `nan` rather than 0 where there are no lines at all.

The null, because a slice always looks like something
-----------------------------------------------------

Cut any set of residuals finely enough and some slice will be two points
off. With 130 teams and a season's worth of games apiece, the most extreme
per-team home bias is *guaranteed* to look interesting, and reading it as a
finding is how a model grows a parameter that fits nothing but the noise it
was fit on.

So every axis is reported against a permutation null of itself: the same
residuals, repointed at the wrong games, the labels reshuffled and the same
dispersion recomputed a few hundred times. That is the shuffle null
`glicko_blend` separates its play-by-play signals at 21 sd with, applied to
a question about slices rather than about games.

`AxisReport.signal_points` is what comes out of it, and it is the number to
make the decision on: the dispersion that survives after the null's own is
taken out, in points of margin. Read it as "the structure on this axis is
worth about this much of a per-game margin correction, at best" -- at best,
because a feature has to find the structure out of sample, and this measures
it in the same games it was computed from. Against football margin MAEs
around 10 (nfl) and 13 (ncaafb), a `signal_points` of 0.1 is not a project.

`sigma` says whether the structure is there at all, `signal_points` says
whether it is worth anything, and the two answer different questions. An
axis with a hundred thousand games behind it can be 8 sd away from its null
and worth 0.06 points, which is a real effect and not a feature. Both
columns are printed because the failure mode of this whole exercise is
reading one of them alone.

What this module does not do
----------------------------

It doesn't slice on anything that isn't already in the predictions frame, so
the axes here are the ones `save_predictions` carries: when the game was
played, who was home, whether the site was neutral, what the model said, and
what the book said. A signal that needs data cassandra has never fetched --
attendance is the standing example, since `endgame.types.Game` has no such
field and never has -- can't be checked here without fetching it first. That
is a feature of the ordering, not a gap in it: the axes that *are* here are
the ones that cost nothing to check, and they are worth exhausting before
anything gets fetched.

And it doesn't tell you a feature will work. A flat axis is strong evidence
against building on it -- there is no error there to remove. A structured
axis is weak evidence for: the structure is real, but whether a model can
predict it *before* the game rather than describe it after is a separate
question and the only one that pays. `control` is the cautionary case, where
a signal with genuine per-game information (21 sd off its shuffle null) was
still worth 1.1% of the gap between two models.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from .columns import GameDfColumns
from .prob_to_margin import (
    BaseProbToMarginFitter,
    MaeLogisticProbToMarginFitter,
)

#: How many relabelings a null is built from. Enough that the null's own
#: standard deviation is estimated to a few percent, which is all `sigma`
#: needs -- the decision it feeds is "is this 1 sd or 10", not a p-value.
DEFAULT_PERMUTATIONS = 400

#: A slice smaller than this is folded into no report. One game's residual is
#: its own noise, and a table with a row per singleton team hides the rows
#: that mean something. Callers that want everything pass `min_games=1`.
DEFAULT_MIN_GAMES = 20

#: Column names this module adds to a predictions frame.
PREDICTED_MARGIN = "predicted_margin"
MARGIN_RESIDUAL = "margin_residual"
MARKET_MARGIN = "market_margin"


class SliceStats(NamedTuple):
    """One slice's worth of residual, in the shape a table row wants.

    `margin_bias` is signed from the home team's side, like the residual it
    averages: positive means home teams beat the model's expectation in this
    slice, so the model is under-rating whatever the slice has in common.

    `win_prob_bias` is `mean(team1_win_prob) - mean(team1_win)`, positive
    when the model was too confident in the home side. It is a second view of
    the same miscalibration on the scale the brier objective sees, and it can
    disagree with `margin_bias` in sign for a slice whose games the model got
    right and blowouts wrong.

    `market_gap` is `margin_mae` minus the market's MAE over `n_lined` games,
    so positive means the market did better. `nan` when the slice has no
    lined games, which is most of NCAAFB's history and every league before
    the odds database starts.
    """

    label: str
    n: int
    win_prob_bias: float
    margin_bias: float
    margin_mae: float
    n_lined: int
    market_gap: float


class AxisReport(NamedTuple):
    """Every slice on one axis, and what the axis is worth as a whole.

    `dispersion` is the game-weighted root mean square of the per-slice
    `margin_bias` *about `overall_bias`*, in points -- one number for "how
    much do these slices disagree with each other". `null_dispersion` is the
    mean of the same statistic over relabelings, which is what pure noise on
    this many slices of these sizes produces.

    `overall_bias` is the mean residual over every game in the report, and it
    is deliberately outside the dispersion. It is a property of the
    prob->margin fit rather than of the axis -- the same number whatever is
    sliced on -- and a fit that runs half a point wide would otherwise show
    up as structure on every axis at once.

    `sigma` is how many null standard deviations the observed dispersion sits
    above the null's mean, and `signal_points` is what is left of the
    dispersion once the null's is removed in quadrature -- the points of real
    per-game structure, at best (see the module docstring on why at best).

    A `sigma` near zero with slices that look interesting is the ordinary
    outcome and the one this class exists to report: the interesting-looking
    slices are what noise does.
    """

    axis: str
    slices: tuple[SliceStats, ...]
    overall_bias: float
    dispersion: float
    null_dispersion: float
    sigma: float
    signal_points: float


class TeamHomeField(NamedTuple):
    """One team's residual split into the part that is home and the part that isn't.

    Both halves are taken from that team's own side of the residual: home
    games as they come, away games negated, so a positive number always means
    the team beat what the model expected of it.

    A team the model simply under-rates is under-rated in both, so the *sum*
    moves and the difference does not. A team with more home advantage than
    the league constant it was given beats expectation at home and doesn't on
    the road, so the *difference* moves. Splitting them is the whole point:
    the first is a rating error the model fixes on its own within a few
    games, and only the second is an argument for a home advantage per team.

    `rating_error` is the half-sum and `home_excess` the whole difference,
    which is the asymmetry that looks arbitrary and isn't. Both are meant to
    come out in the units of the thing they describe, and the two things are
    on different scales.

    A rating error of `e` shifts both halves by `e`, so their mean recovers
    `e` and the half is right. A home edge of `d` points more than the league
    constant does something else: the ratings are fit on a schedule that is
    half home games, so a converged replay has already absorbed `d/2` of it
    into the team's rating, leaving `+d/2` at home and `-d/2` away. The
    difference of those is `d`, and halving it would report every team's home
    edge at half its size. So `home_excess` of 1.5 means this team's home
    advantage is about 1.5 points above the league's fitted constant, which
    is the number a per-team parameter would have to hold.

    The absorption is what makes the two columns separate cleanly, and it is
    an assumption about a converged replay rather than an identity. A team
    that played almost all of one season at home, or one whose rating never
    settled, splits less tidily -- `rating_error` picks up some of the home
    edge and the difference stays right.

    Neutral-site games are in neither: the model applies no home advantage
    there, so they carry no information about how big it should have been,
    and counting them as home games dilutes the estimate by however much of a
    schedule is bowls.
    """

    team: str
    home_games: int
    away_games: int
    rating_error: float
    home_excess: float


def add_residuals(
    df: pd.DataFrame, fitter: BaseProbToMarginFitter | None = None
) -> pd.DataFrame:
    """The predictions frame with the margin the model implied, and its error.

    One global prob->margin fit over every game in `df`, exactly as
    `score_predictions` and the `margin_mae` objective do it, then the
    residual off that fit. Slicing happens downstream, on the frame this
    returns -- see the module docstring for why the fit can't move with it.

    Defaults to the MAE logistic because that is what the football configs
    optimize against, so the residuals here are the ones the search was
    charged for. Pass another fitter to ask the question of a model tuned on
    a different objective.
    """
    if df.empty:
        # The same guard the rest of the scoring path has, for the same
        # reason: the assigns below reach for columns an empty frame doesn't
        # have, and the AttributeError reads like a schema bug.
        raise ValueError("No games to score")
    fitter = fitter or MaeLogisticProbToMarginFitter()
    games = df.assign(
        **{GameDfColumns.TEAM1_MOV: lambda x: x.home_score - x.away_score}
    )
    margin_predictor = fitter.fit_df(games)
    predicted = margin_predictor.predict_margins(
        games[GameDfColumns.TEAM1_WIN_PROB].to_numpy()
    )
    return games.assign(
        **{
            PREDICTED_MARGIN: predicted,
            MARGIN_RESIDUAL: games[GameDfColumns.TEAM1_MOV].to_numpy() - predicted,
            # A spread is quoted from team1's side and team1 covers when
            # spread + mov > 0, so the margin it implies is its negation --
            # the same sign `score_predictions` takes, and the same one that
            # produces a plausible-looking MAE if it's flipped.
            #
            # Coerced first because `spread` is `float | None`, and a league
            # the odds database has never quoted -- nfl, as of this writing --
            # arrives as a column of `None` that pandas types as `object`.
            # Negating that raises `bad operand type for unary -: 'NoneType'`,
            # which is a crash rather than the empty `market_gap` the reports
            # already know how to print. Mixed None-and-float columns come
            # back as float64 with NaN on their own, so this only bites the
            # all-missing case, which is exactly the one a test frame written
            # by hand doesn't reproduce.
            MARKET_MARGIN: -pd.to_numeric(games[GameDfColumns.SPREAD], errors="coerce"),
        }
    )


def _slice_stats(label: str, rows: pd.DataFrame) -> SliceStats:
    residual = rows[MARGIN_RESIDUAL].to_numpy()
    lined = rows[rows[MARKET_MARGIN].notna()]
    if lined.empty:
        market_gap = float("nan")
    else:
        model_mae = np.abs(lined[MARGIN_RESIDUAL].to_numpy()).mean()
        market_mae = np.abs(
            lined[MARKET_MARGIN].to_numpy() - lined[GameDfColumns.TEAM1_MOV].to_numpy()
        ).mean()
        market_gap = float(model_mae - market_mae)
    return SliceStats(
        label=label,
        n=len(rows),
        win_prob_bias=float(
            rows[GameDfColumns.TEAM1_WIN_PROB].mean()
            - rows[GameDfColumns.TEAM1_WIN].astype(float).mean()
        ),
        margin_bias=float(residual.mean()),
        margin_mae=float(np.abs(residual).mean()),
        n_lined=len(lined),
        market_gap=market_gap,
    )


def _dispersion(residual: np.ndarray, codes: np.ndarray, n_slices: int) -> float:
    """Game-weighted RMS of the per-slice mean residual, about the grand mean.

    About the grand mean rather than about zero, which is the whole
    difference between an axis statistic and a calibration one. A model whose
    residuals average +0.8 points overall is 0.8 off in *every* slice, and
    measuring dispersion about zero would report that shared offset as though
    the slices disagreed with each other. They don't: a global bias is an
    argument for refitting the prob->margin scale, not for a feature. What
    this measures is how far the slices are from each other.

    Weighted by slice size rather than flat, so a twenty-game slice's noisy
    mean can't dominate a statistic meant to describe the whole axis -- and
    so the number stays in points of the per-game correction it would take to
    flatten the axis out.

    `np.bincount` rather than a groupby: this runs once per permutation, a
    few hundred times per axis, and a pandas groupby at that rate is most of
    the module's runtime.
    """
    counts = np.bincount(codes, minlength=n_slices)
    sums = np.bincount(codes, weights=residual, minlength=n_slices)
    present = counts > 0
    means = sums[present] / counts[present]
    total = counts[present].sum()
    grand = sums[present].sum() / total
    return float(np.sqrt((counts[present] * (means - grand) ** 2).sum() / total))


def _signal_points(observed: float, null: np.ndarray) -> float:
    """What is left of `observed` once the null's own dispersion is removed.

    In quadrature, because the observed dispersion is structure and noise
    added as variances rather than as points -- a slice mean is the real
    per-slice bias plus a sampling error, and the two are independent.

    Against the null's *root mean square* rather than its mean. Those differ:
    the null dispersion is a positive quantity with spread, so
    `mean(D)^2 < mean(D^2)`, and subtracting the smaller of the two would
    leave a sliver of the noise behind and report it as signal. It is a small
    difference on a well-sampled axis and not a small one on an axis with a
    handful of slices, which is exactly where a spurious tenth of a point
    would be believed.

    `null_dispersion` on the report stays the mean, because that is the
    number to read next to `dispersion` -- "this is what noise alone
    produces" -- and the RMS is a hair higher for no reason a reader would
    guess.

    Floored at 0: an axis flatter than its own null is an axis with nothing
    on it, not one with negative structure.
    """
    return float(np.sqrt(max(0.0, observed**2 - float((null**2).mean()))))


def axis_report(
    df: pd.DataFrame,
    labels: pd.Series,
    axis: str,
    min_games: int = DEFAULT_MIN_GAMES,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> AxisReport:
    """Slice `df`'s residuals by `labels` and say whether the axis carries anything.

    `df` must have been through `add_residuals`. `labels` is one value per
    row, aligned on the index -- whatever the axis is, already computed. The
    slicing is deliberately left to the caller: a week number, a bucket of
    them, a team name and a favorite size are all the same operation once
    somebody has decided what the buckets are, and the deciding is the part
    worth reading in the calling code.

    Slices under `min_games` are dropped from the table *and* from the null,
    so the two are computed over the same games. Dropping them from only the
    first is how an axis picks up sigma from rows nobody reports.

    `seed` is fixed by default: the null is a measurement like the others,
    and one that moves between two runs of the same frame invites rerunning
    it until it agrees.
    """
    labeled = df.assign(_axis_label=labels.astype(str).to_numpy())
    sizes = labeled["_axis_label"].value_counts()
    kept = set(sizes[sizes >= min_games].index)
    labeled = labeled[labeled["_axis_label"].isin(kept)]
    if labeled.empty or len(kept) < 2:
        # One slice has nothing to disagree with, and no slices have nothing
        # at all. Reporting zeros rather than raising: a caller sweeping
        # every axis over a small league shouldn't have to know in advance
        # which ones its schedule collapses.
        return AxisReport(
            axis=axis,
            slices=tuple(
                _slice_stats(label, rows)
                for label, rows in labeled.groupby("_axis_label")
            ),
            overall_bias=(
                float(labeled[MARGIN_RESIDUAL].mean()) if not labeled.empty else 0.0
            ),
            dispersion=0.0,
            null_dispersion=0.0,
            sigma=0.0,
            signal_points=0.0,
        )

    # groupby already yields the labels in sorted order, which is the order
    # a week table wants and the only one a caller can predict.
    slices = tuple(
        _slice_stats(label, rows) for label, rows in labeled.groupby("_axis_label")
    )
    codes, _ = pd.factorize(labeled["_axis_label"])
    residual = labeled[MARGIN_RESIDUAL].to_numpy()
    observed = _dispersion(residual, codes, len(kept))

    rng = np.random.default_rng(seed)
    # Shuffling the residuals rather than the labels, which is the same null
    # and one array copy instead of a re-factorize.
    null = np.array(
        [
            _dispersion(rng.permutation(residual), codes, len(kept))
            for _ in range(permutations)
        ]
    )
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    return AxisReport(
        axis=axis,
        slices=slices,
        overall_bias=float(residual.mean()),
        dispersion=observed,
        null_dispersion=null_mean,
        # A null with no spread at all means every permutation landed on the
        # same number, which happens only for degenerate input; 0 sigma is
        # the honest reading of it rather than an inf.
        sigma=0.0 if null_sd == 0 else (observed - null_mean) / null_sd,
        signal_points=_signal_points(observed, null),
    )


def home_field_table(
    df: pd.DataFrame, min_games: int = DEFAULT_MIN_GAMES
) -> tuple[TeamHomeField, ...]:
    """Every team's residual split into a rating half and a home-field half.

    `df` must have been through `add_residuals`, and must carry
    `neutral_site` -- neutral games are excluded, since the model gives
    nobody a home advantage in one and they say nothing about how big it
    should be.

    `min_games` applies to each side separately: a team needs that many home
    games *and* that many away ones, because `home_excess` is a difference
    and a difference is only as good as its thinner half.

    Sorted by `home_excess`, descending, so the teams the league constant
    under-serves are at the top. Read the table against
    `home_field_report`'s null before reading any single row -- with a
    hundred-odd teams the extremes of a pure-noise table are large.
    """
    if "neutral_site" not in df.columns:
        raise ValueError(
            "home_field_table needs a `neutral_site` column; re-run the "
            "predictions through save_predictions to get one"
        )
    sided = df[~df["neutral_site"].astype(bool)]
    # One row per team per game, with the residual flipped to that team's
    # side, so a groupby does both halves at once.
    home = sided[["home_team", MARGIN_RESIDUAL]].rename(columns={"home_team": "team"})
    away = sided[["away_team", MARGIN_RESIDUAL]].rename(columns={"away_team": "team"})
    away = away.assign(**{MARGIN_RESIDUAL: -away[MARGIN_RESIDUAL]})
    home_means = home.groupby("team")[MARGIN_RESIDUAL].agg(["mean", "size"])
    away_means = away.groupby("team")[MARGIN_RESIDUAL].agg(["mean", "size"])
    joined = home_means.join(away_means, how="inner", lsuffix="_home", rsuffix="_away")
    joined = joined[
        (joined["size_home"] >= min_games) & (joined["size_away"] >= min_games)
    ]
    table = [
        TeamHomeField(
            team=str(team),
            home_games=int(row["size_home"]),
            away_games=int(row["size_away"]),
            rating_error=float((row["mean_home"] + row["mean_away"]) / 2),
            home_excess=float(row["mean_home"] - row["mean_away"]),
        )
        for team, row in joined.iterrows()
    ]
    return tuple(sorted(table, key=lambda t: t.home_excess, reverse=True))


def home_field_report(
    df: pd.DataFrame,
    min_games: int = DEFAULT_MIN_GAMES,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> AxisReport:
    """Whether home advantage varying by team is a real thing in this league.

    The statistic is the game-weighted RMS of `TeamHomeField.home_excess`
    about its own mean -- how far apart teams' home edges are, in points --
    against a null that permutes the residuals across games and recomputes
    it. The null is what makes this readable: teams differ in their measured
    home edge under *any* shuffle, because each estimate is a difference of
    two noisy means, and the question is only whether they differ by more
    than that.

    About the mean rather than about zero, so a league whose fitted
    `home_advantage` is uniformly too small doesn't read as a case for a
    parameter per team. That case is `overall_bias` on the sidelines and one
    number in the search's range, not 130 of them.

    `signal_points` here has a direct reading the other axes don't have. It
    is roughly the standard deviation of the per-team home advantage around
    the league constant, in points, so it is the size of the parameter a
    per-team home advantage would be estimating. A value well under a point
    against a league fit at 3.5 says the constant is close enough and a
    parameter per team would be fitting its own noise.

    What it cannot separate is home advantage from anything else that lives
    at the same address: altitude, turf, travel that always runs one
    direction, a program that is simply better than its rating at home for
    reasons nobody has named. It reports that the split exists and leaves
    what causes it open -- which is fine for the decision it feeds, since a
    per-team home advantage would absorb all of them equally.
    """
    table = home_field_table(df, min_games=min_games)
    slices = tuple(
        SliceStats(
            label=t.team,
            n=t.home_games + t.away_games,
            # A team's row carries its home edge and its rating error and
            # nothing else. The three columns a slice on an ordinary axis
            # fills are about a set of games; these are about a team, and a
            # number that looked like a per-team margin MAE would invite
            # being read as one.
            win_prob_bias=t.rating_error,
            margin_bias=t.home_excess,
            margin_mae=float("nan"),
            n_lined=0,
            market_gap=float("nan"),
        )
        for t in table
    )
    sided = df[~df["neutral_site"].astype(bool)]
    # One index over everybody who appears on either side, so a team's home
    # code and its away code are the same number and the min_games mask below
    # is the same one `home_field_table` applied. Observed and null then come
    # out of one function, which is what makes the sigma mean anything.
    teams = pd.Index(pd.unique(pd.concat([sided["home_team"], sided["away_team"]])))
    home_codes = teams.get_indexer(sided["home_team"])
    away_codes = teams.get_indexer(sided["away_team"])
    residual = sided[MARGIN_RESIDUAL].to_numpy()
    n_teams = len(teams)
    home_n = np.bincount(home_codes, minlength=n_teams)
    away_n = np.bincount(away_codes, minlength=n_teams)
    both = (home_n >= min_games) & (away_n >= min_games)

    def _excess_dispersion(values: np.ndarray) -> float:
        home_sum = np.bincount(home_codes, weights=values, minlength=n_teams)
        away_sum = np.bincount(away_codes, weights=-values, minlength=n_teams)
        excess = home_sum[both] / home_n[both] - away_sum[both] / away_n[both]
        counts = (home_n + away_n)[both].astype(float)
        # About the weighted mean, for the reason `_dispersion` centers: a
        # league whose home advantage constant is simply too small shifts
        # every team's excess by the same amount, and that is an argument for
        # refitting the constant rather than for a parameter per team.
        centre = (counts * excess).sum() / counts.sum()
        return float(np.sqrt((counts * (excess - centre) ** 2).sum() / counts.sum()))

    if both.sum() < 2:
        return AxisReport(
            axis="home_field",
            slices=slices,
            overall_bias=float(residual.mean()) if len(residual) else 0.0,
            dispersion=0.0,
            null_dispersion=0.0,
            sigma=0.0,
            signal_points=0.0,
        )
    observed = _excess_dispersion(residual)

    rng = np.random.default_rng(seed)
    null = np.array(
        [_excess_dispersion(rng.permutation(residual)) for _ in range(permutations)]
    )
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))
    return AxisReport(
        axis="home_field",
        slices=slices,
        overall_bias=float(residual.mean()),
        dispersion=observed,
        null_dispersion=null_mean,
        sigma=0.0 if null_sd == 0 else (observed - null_mean) / null_sd,
        signal_points=_signal_points(observed, null),
    )


def season_stage(df: pd.DataFrame, early: int = 4, late: int = 12) -> pd.Series:
    """Week numbers folded into early / middle / late.

    The coarse companion to slicing on `week_number` itself. Both are worth
    running: the fine one shows the shape and the coarse one has the sample
    to say whether the shape is there, and an effect that is real but small
    shows up as sigma on this one and noise on that one.

    The cuts are the football ones. Through week 4 is the stretch a rating
    model is still working off last season's ratings, which is where anything
    known about the offseason -- who transferred, what a recruiting class
    was worth -- would have to pay for itself. From week 12 on is conference
    championships and the postseason, where the schedule stops being the one
    the ratings were built on.
    """
    week = df["week_number"]
    return pd.Series(
        np.where(week <= early, "early", np.where(week >= late, "late", "middle")),
        index=df.index,
    )


def favorite_size(
    df: pd.DataFrame, edges: Sequence[float] = (3, 7, 14, 21)
) -> pd.Series:
    """The model's own predicted margin, bucketed by size.

    An axis about the model rather than about the games: it asks whether the
    prob->margin mapping is right across its range, which is the one thing a
    single global logistic scale genuinely cannot represent if it isn't. A
    bias that grows with the bucket is a scale error; one that flips sign at
    the extremes is a shape error, and `IsotonicProbToMarginFitter` is
    already in the tree for that case.

    Signed, so the home and away halves are separate buckets: a mapping that
    is right for home favorites and wrong for away ones is a home advantage
    problem wearing a calibration costume, and folding the sign away would
    hide it.
    """
    return pd.cut(
        df[PREDICTED_MARGIN],
        bins=[-np.inf, *(-e for e in reversed(edges)), *edges, np.inf],
    ).astype(str)


def rest_advantage(df: pd.DataFrame, edges: Sequence[float] = (2, 5)) -> pd.Series:
    """How many more days off the home team had than the away team, bucketed.

    Off the game dates and nothing else, which is what makes it worth
    running before anything gets fetched: every model already replays a
    frame that carries `date`, `home_team` and `away_team`, so this axis
    costs one groupby and no new data at all.

    None of the rating models can see it. A team is one number, and that
    number is the same whether it last played six days ago or thirteen -- so
    a bye week, a Thursday game on four days, and a bowl on a month's rest
    are all priced identically. If rest matters, it is entirely in the
    residual, which is the ideal shape for this diagnostic: no part of the
    effect has been absorbed already.

    A team's first game of a replay has no previous game to count from, and
    a team's first game of a *season* counts from the last one of the season
    before -- an offseason, not a rest advantage. Both land in `unknown`
    rather than in a bucket, since a 200-day gap in a rest table is the kind
    of row that gets read as a finding.
    """
    dates = pd.to_datetime(df["date"]).to_numpy()
    # Keyed by position rather than by the frame's index, which a caller who
    # sliced a frame before handing it over may have left with duplicates --
    # and a duplicated index turns the realignment below into a cross join
    # that reports a rest table for many times the games there are.
    position = np.arange(len(df))
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "team": df["home_team"].to_numpy(),
                    "date": dates,
                    "side": "home",
                    "position": position,
                }
            ),
            pd.DataFrame(
                {
                    "team": df["away_team"].to_numpy(),
                    "date": dates,
                    "side": "away",
                    "position": position,
                }
            ),
        ],
        ignore_index=True,
    )
    long = long.sort_values("date", kind="stable")
    long["rest"] = long.groupby("team")["date"].diff().dt.days
    sides = {
        side: rows.set_index("position")["rest"].reindex(position).to_numpy()
        for side, rows in long.groupby("side")
    }
    home_rest = pd.Series(sides["home"], index=df.index)
    away_rest = pd.Series(sides["away"], index=df.index)
    # Anything over a month apart is a season boundary rather than a bye, and
    # the offseason is `season_stage`'s question, not this one.
    offseason = (home_rest > 40) | (away_rest > 40)
    difference = (home_rest - away_rest).where(~offseason)
    bucketed = pd.cut(
        difference, bins=[-np.inf, *(-e for e in reversed(edges)), *edges, np.inf]
    ).astype(str)
    return bucketed.where(difference.notna(), "unknown")


def standard_axes(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """The axes worth running on any football model, ready for `axis_report`.

    Everything computable from the predictions frame alone, which is the
    boundary this module keeps: no sweep, no second data source, no refit.
    A caller with more to slice on -- a conference from
    `call_it_what_you_want`, days of rest off the game dates, EPA coverage
    off the index -- adds it here or passes its own labels.
    """
    return {
        "week": df["week_number"].astype(str),
        "season_stage": season_stage(df),
        "year": df["year"].astype(str),
        "favorite_size": favorite_size(df),
        "rest_advantage": rest_advantage(df),
        # Not the home-field question -- `home_field_report` is that, and it
        # nets a team's home games against its away ones. This one is flatter
        # and asks whether some teams' games are harder to call than others
        # at all, which a rating model has no parameter for either way.
        "home_team": df["home_team"],
    }
