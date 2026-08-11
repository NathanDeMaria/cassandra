import json

import numpy as np
import pandas as pd
import pytest

from .base_fit import BaseProbToMarginPredictor
from .isotonic import IsotonicProbToMarginFitter, IsotonicProbToMarginPredictor


def _fitted() -> BaseProbToMarginPredictor:
    """Fit the way production does: through fit_df, which doubles the data.

    fit_df mirrors every game onto the away side (mov and win prob both
    flipped), so fitting through `fit` directly would exercise a shape the
    real caller never produces.
    """
    rng = np.random.default_rng(0)
    win_probs = rng.uniform(0.1, 0.9, 200)
    games = pd.DataFrame(
        {
            "team1_win_prob": win_probs,
            "team1_mov": 30 * (win_probs - 0.5) + rng.normal(0, 7, 200),
        }
    )
    return IsotonicProbToMarginFitter().fit_df(games)


def test_round_trips_through_a_dict() -> None:
    predictor = _fitted()
    win_probs = np.linspace(0.05, 0.95, 50)

    rehydrated = BaseProbToMarginPredictor.from_dict(predictor.to_dict())

    assert rehydrated.predict_margins(win_probs) == pytest.approx(
        predictor.predict_margins(win_probs)
    )


def test_round_trips_through_json() -> None:
    """The dict has to survive an actual serialize, not just look like data.

    np.float64 clears json.dumps only because it subclasses float; the rest
    of the numpy float family raises, so the knots get cast on the way out.
    """
    predictor = _fitted()

    payload = json.dumps(predictor.to_dict())
    rehydrated = BaseProbToMarginPredictor.from_dict(json.loads(payload))

    win_probs = np.linspace(0.05, 0.95, 50)
    assert rehydrated.predict_margins(win_probs) == pytest.approx(
        predictor.predict_margins(win_probs)
    )


def test_serialized_shape_is_what_the_consumer_pins() -> None:
    data = _fitted().to_dict()

    assert data.keys() == {"kind", "x_thresholds", "y_thresholds"}
    assert data["kind"] == "isotonic"
    assert len(data["x_thresholds"]) == len(data["y_thresholds"])
    # Plain builtins, not numpy scalars: a consumer feeds these straight to
    # np.interp, which needs ascending x and nothing more.
    assert all(type(x) is float for x in data["x_thresholds"])
    assert all(type(y) is float for y in data["y_thresholds"])
    assert data["x_thresholds"] == sorted(data["x_thresholds"])


def test_rehydrates_without_refitting_or_sklearn() -> None:
    """A consumer holds the knots and nothing else -- no fitter, no estimator.

    This is the whole point: the web service builds a predictor from a saved
    dict it read off disk, never having seen the games it was fit on.
    """
    data = {
        "kind": "isotonic",
        "x_thresholds": [0.2, 0.5, 0.8],
        "y_thresholds": [-10.0, 0.0, 10.0],
    }

    predictor = BaseProbToMarginPredictor.from_dict(data)

    assert isinstance(predictor, IsotonicProbToMarginPredictor)
    assert predictor.predict_margins(np.array([0.5])) == pytest.approx(0.0)
    assert predictor.predict_margins(np.array([0.35])) == pytest.approx(-5.0)


def test_monotonic_in_win_probability() -> None:
    predictor = _fitted()

    predicted = predictor.predict_margins(np.linspace(0.05, 0.95, 100))

    # Isotonic output is a step function, so equal neighbours are expected --
    # what can't happen is a higher win prob implying a smaller margin.
    assert np.all(np.diff(predicted) >= 0)
    assert predicted[-1] > predicted[0]


def test_a_lopsided_matchup_clamps_instead_of_going_nan() -> None:
    """The trap: IsotonicRegression.predict returns nan outside the fit.

    The fit only spans the win probs the season contained, so a 99% favourite
    falls off the end. nan would travel all the way to a blank number on a
    web page, so np.interp's clamping is what the consumer gets.
    """
    predictor = _fitted()

    extreme = predictor.predict_margins(np.array([0.001, 0.999]))

    assert not np.any(np.isnan(extreme))
    in_range = predictor.predict_margins(np.array([0.1, 0.9]))
    assert extreme[0] == pytest.approx(in_range[0])
    assert extreme[1] == pytest.approx(in_range[1])


def test_clamps_at_the_end_knots() -> None:
    predictor = IsotonicProbToMarginPredictor.from_dict(
        {
            "kind": "isotonic",
            "x_thresholds": [0.2, 0.8],
            "y_thresholds": [-10.0, 10.0],
        }
    )

    predicted = predictor.predict_margins(np.array([0.0, 0.1, 0.9, 1.0]))

    assert predicted == pytest.approx([-10.0, -10.0, 10.0, 10.0])


def test_matches_the_sklearn_fit_it_came_from() -> None:
    """The knots are only worth keeping if they reproduce the estimator.

    Everything else rests on np.interp over (X_thresholds_, y_thresholds_)
    agreeing with IsotonicRegression.predict inside the fitted range.
    """
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(1)
    win_probs = rng.uniform(0.1, 0.9, 300)
    margins = 30 * (win_probs - 0.5) + rng.normal(0, 7, 300)
    iso = IsotonicRegression(increasing=True).fit(win_probs, margins)
    predictor = IsotonicProbToMarginPredictor(iso.X_thresholds_, iso.y_thresholds_)

    inside = np.linspace(iso.X_min_, iso.X_max_, 500)

    assert predictor.predict_margins(inside) == pytest.approx(iso.predict(inside))


def test_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown prob-to-margin predictor kind"):
        BaseProbToMarginPredictor.from_dict({"kind": "quadratic"})
