import json

import numpy as np
import pytest

from .base_fit import BaseProbToMarginPredictor
from .isotonic import IsotonicProbToMarginFitter
from .logistic import LogisticProbToMarginFitter, _logit


def test_pick_em_maps_to_zero_margin() -> None:
    predictor = LogisticProbToMarginFitter().fit(
        win_probs=np.array([0.5, 0.7, 0.3]), margins=np.array([0.0, 3.0, -3.0])
    )
    assert predictor.predict_margins(np.array([0.5])) == pytest.approx(0.0)


def test_recovers_known_scale() -> None:
    true_scale = 12.0
    win_probs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    margins = true_scale * _logit(win_probs)

    predictor = LogisticProbToMarginFitter().fit(win_probs, margins)

    predicted = predictor.predict_margins(win_probs)
    assert predicted == pytest.approx(margins)


def test_monotonic_increasing() -> None:
    predictor = LogisticProbToMarginFitter().fit(
        win_probs=np.array([0.5, 0.7, 0.9]), margins=np.array([0.0, 4.0, 10.0])
    )
    win_probs = np.linspace(0.05, 0.95, 20)
    predicted = predictor.predict_margins(win_probs)
    assert np.all(np.diff(predicted) > 0)


def test_averages_out_the_noise_in_a_single_game() -> None:
    """Margins are noisy in a way spreads aren't, so the fit has to smooth.

    Each of these pick'em games finished nowhere near level, but they cancel,
    and the fit says the next one is a coin flip rather than chasing the last
    result. This is the cost of the extra training data, and the reason the
    target is a least-squares fit and not an interpolation.
    """
    win_probs = np.array([0.5, 0.5, 0.7, 0.7])
    margins = np.array([21.0, -21.0, 14.0, -8.0])

    predictor = LogisticProbToMarginFitter().fit(win_probs, margins)

    assert predictor.predict_margins(np.array([0.5])) == pytest.approx(0.0)
    predicted = predictor.predict_margins(np.array([0.7]))[0]
    assert 0.0 < predicted < 14.0


def test_round_trips_through_json() -> None:
    """The other fitter has to be serializable too, or it can't be deployed.

    A least-squares scale is an np.float64, which json.dumps only tolerates
    because it subclasses float -- cast on the way out rather than rely on it.
    """
    predictor = LogisticProbToMarginFitter().fit(
        win_probs=np.array([0.5, 0.7, 0.9]), margins=np.array([0.0, 4.0, 10.0])
    )

    rehydrated = BaseProbToMarginPredictor.from_dict(
        json.loads(json.dumps(predictor.to_dict()))
    )

    win_probs = np.linspace(0.05, 0.95, 20)
    assert rehydrated.predict_margins(win_probs) == pytest.approx(
        predictor.predict_margins(win_probs)
    )


def test_extrapolates_past_observed_range_unlike_isotonic() -> None:
    # Isotonic regression can't say anything past the most extreme win_prob
    # it saw during fitting -- it just flattens out at the boundary value.
    # The logistic fit keeps extrapolating a bigger margin for a bigger
    # mismatch, which is what you actually want for a lopsided matchup.
    win_probs = np.array([0.4, 0.5, 0.6])
    margins = np.array([-2.0, 0.0, 2.0])

    isotonic_predictor = IsotonicProbToMarginFitter().fit(win_probs, margins)
    logistic_predictor = LogisticProbToMarginFitter().fit(win_probs, margins)

    extreme_win_prob = np.array([0.999])
    isotonic_margin = isotonic_predictor.predict_margins(extreme_win_prob)[0]
    logistic_margin = logistic_predictor.predict_margins(extreme_win_prob)[0]

    # Isotonic regression has nothing useful to say past the data it saw, so
    # it clamps at the most extreme margin it observed, while the logistic
    # fit keeps extrapolating a bigger margin.
    assert isotonic_margin == pytest.approx(2.0)
    assert logistic_margin > 10.0
