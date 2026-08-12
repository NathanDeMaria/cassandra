"""The ModelRelease artifact: the contract between cassandra and its consumers.

A release is a self-contained snapshot of one model -- its ratings, the fit
that turns a win probability into a margin, and the metrics that say how good
it is. Consumers read the JSON and never run the model, which is what lets a
web service answer "what margin does a 62% win probability imply?" without
sklearn, without the season pickles, and without cassandra's fitting stack.

This module owns the schema so there is one definition rather than two that
drift. It deliberately depends on nothing heavier than pydantic,
`cassandra.prob_to_margin` and `cassandra.predictor` -- none of which import
scikit-learn at module scope, which is the property that lets a consumer
install cassandra without the `fit` group and still read a release.
"""

import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from cassandra.predictor import Predictor, Rating, load_predictor_class
from cassandra.prob_to_margin import BaseProbToMarginPredictor


class TeamRating(BaseModel):
    """A team's standing in one release.

    `rd` is Glicko's rating deviation; Elo and Elo538 don't have one, so it
    stays None rather than being faked as 0.
    """

    rating: float
    rd: float | None = None
    wins: int = 0
    losses: int = 0


def ratings_from_predictor(predictor: Predictor) -> dict[str, TeamRating]:
    """Snapshot a predictor's ratings into their release form.

    Win/loss records aren't the predictor's to know -- they come from the
    games -- so they stay at 0 here and a caller that has them fills them in.

    Raises RatingsUnsupported for a predictor with no ratings (FlatPredictor).
    """
    return {
        team: TeamRating(rating=rating.rating, rd=rating.rd)
        for team, rating in predictor.ratings.items()
    }


class _MarginCalibrationBase(BaseModel):
    """Shared rehydration for every calibration kind.

    The serialized form is exactly what `BaseProbToMarginPredictor.to_dict`
    emits, so the pydantic model and the predictor never disagree about the
    payload -- there's one definition of it, in prob_to_margin, and this
    validates that same dict.
    """

    def to_predictor(self) -> BaseProbToMarginPredictor:
        """Rebuild the predictor. No fitter, no scikit-learn, no refit."""
        return BaseProbToMarginPredictor.from_dict(self.model_dump())


class IsotonicMarginCalibration(_MarginCalibrationBase):
    """A fitted prob->margin mapping, stored as isotonic regression knots.

    Knots rather than a pickled sklearn estimator so serving needs only
    np.interp, and so a scikit-learn upgrade can't make old releases
    unreadable.

    Both lists ascend: the fit is `IsotonicRegression(increasing=True)` over
    margin of victory, so a higher win probability means the home team wins by
    more. np.interp needs ascending x and clamps outside the fitted range,
    which is the intended behavior for a matchup more lopsided than anything
    the season contained.

    Note the sign: this is *margin* (positive = home wins by that much), not
    spread. A consumer rendering "Duke -6.5" negates it.
    """

    kind: Literal["isotonic"]
    x_thresholds: list[float]
    y_thresholds: list[float]


class LogisticMarginCalibration(_MarginCalibrationBase):
    """`margin = scale * logit(win_prob)`.

    The smooth alternative to isotonic: it keeps extrapolating past the win
    probabilities the season actually contained instead of flattening at the
    most extreme margin observed, which is the difference that shows up on a
    lopsided matchup.
    """

    kind: Literal["logistic"]
    scale: float


MarginCalibration = Annotated[
    IsotonicMarginCalibration | LogisticMarginCalibration,
    Field(discriminator="kind"),
]

_CALIBRATION_ADAPTER: TypeAdapter[MarginCalibration] = TypeAdapter(MarginCalibration)


def calibration_from_predictor(
    predictor: BaseProbToMarginPredictor,
) -> MarginCalibration:
    """Serialize a fitted predictor into its release form."""
    return _CALIBRATION_ADAPTER.validate_python(predictor.to_dict())


class Metrics(BaseModel):
    """Evaluation metrics for a release.

    `brier_score` and the MAEs are error measures, so lower is better -- a
    consumer picking a default model wants a min, not a max.

    The optional fields are None for a league the odds database doesn't cover
    at all. They're None rather than nan on purpose: `score_predictions`
    returns nan there, and nan survives json.dumps as the literal `NaN`, which
    is not valid JSON and which browsers reject.
    """

    brier_score: float
    margin_mae: float
    against_spread_accuracy: float | None = None
    spread_game_margin_mae: float | None = None
    market_margin_mae: float | None = None
    n_games: int
    n_spread_games: int = 0


def _no_nan(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def metrics_from_scored(metrics: dict[str, float]) -> Metrics:
    """Build Metrics from a `ScoredPredictions.metrics` dict.

    The nan-to-None mapping is the whole reason this exists; see Metrics.
    """
    return Metrics(
        brier_score=float(metrics["brier_score"]),
        margin_mae=float(metrics["margin_mae"]),
        against_spread_accuracy=_no_nan(metrics["against_spread_accuracy"]),
        spread_game_margin_mae=_no_nan(metrics["spread_game_margin_mae"]),
        market_margin_mae=_no_nan(metrics["market_margin_mae"]),
        n_games=int(metrics["n_games"]),
        n_spread_games=int(metrics["n_spread_games"]),
    )


class TrainedThrough(BaseModel):
    """Watermark for the incremental refresh job.

    `processed_game_ids` is a set of ids rather than a timestamp because games
    get re-fetched and scores corrected; ids are what make the refresh
    idempotent. Current season only -- it's a resume marker, not an audit log.
    """

    season_year: int
    last_game_date: datetime | None = None
    processed_game_ids: list[str] = Field(default_factory=list)


class ModelRelease(BaseModel):
    """One model's ratings and calibration at a point in time."""

    # `model` collides with pydantic's protected `model_` namespace warning
    # even though it isn't itself a reserved attribute.
    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal[1] = 1
    run_id: str
    league: str
    model: str
    predictor_class: str
    params: dict[str, float | str] = Field(default_factory=dict)
    ratings: dict[str, TeamRating] = Field(default_factory=dict)
    margin_calibration: MarginCalibration | None = None
    metrics: Metrics
    trained_through: TrainedThrough
    created_at: datetime
    created_by: str
    parent_run_id: str | None = None

    def margin_predictor(self) -> BaseProbToMarginPredictor | None:
        """The calibration as something you can call, or None if unfit."""
        if self.margin_calibration is None:
            return None
        return self.margin_calibration.to_predictor()

    def rating_predictor(self) -> Predictor:
        """The rating model, rebuilt from `predictor_class`, params and ratings.

        Named for what it returns rather than `to_predictor`, because a
        release rehydrates two different predictors and a consumer holding
        both wants to see which is which at the call site.

        No state file and no temp file: the release already carries everything
        the constructor needs, so the denormalizing happens in the predictor
        class that knows its own rating shape.

        Raises UnknownPredictorClass if this build has no such class -- the
        expected outcome for a release written by a newer cassandra -- and
        RatingsUnsupported for a class that doesn't rate teams at all.
        """
        predictor_class = load_predictor_class(self.predictor_class)
        return predictor_class.from_ratings(
            self.league,
            {
                team: Rating(rating=rating.rating, rd=rating.rd)
                for team, rating in self.ratings.items()
            },
            **self.params,
        )


def release_json_schema() -> dict[str, Any]:
    """The JSON Schema for a release, for consumers checking their fixtures."""
    return ModelRelease.model_json_schema()
