import numpy as np
import pytest

from .isotonic import IsotonicProbToSpreadFitter
from .logistic import LogisticProbToSpreadFitter, _logit


def test_pick_em_maps_to_zero_spread() -> None:
    predictor = LogisticProbToSpreadFitter().fit(
        win_probs=np.array([0.5, 0.7, 0.3]), spreads=np.array([0.0, -3.0, 3.0])
    )
    assert predictor.predict_spreads(np.array([0.5])) == pytest.approx(0.0)


def test_recovers_known_scale() -> None:
    true_scale = 12.0
    win_probs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    spreads = -true_scale * _logit(win_probs)

    predictor = LogisticProbToSpreadFitter().fit(win_probs, spreads)

    predicted = predictor.predict_spreads(win_probs)
    assert predicted == pytest.approx(spreads)


def test_monotonic_decreasing() -> None:
    predictor = LogisticProbToSpreadFitter().fit(
        win_probs=np.array([0.5, 0.7, 0.9]), spreads=np.array([0.0, -4.0, -10.0])
    )
    win_probs = np.linspace(0.05, 0.95, 20)
    predicted = predictor.predict_spreads(win_probs)
    assert np.all(np.diff(predicted) < 0)


def test_extrapolates_past_observed_range_unlike_isotonic() -> None:
    # Isotonic regression can't say anything past the most extreme win_prob
    # it saw during fitting -- it just flattens out at the boundary value.
    # The logistic fit keeps extrapolating a bigger spread for a bigger
    # mismatch, which is what you actually want for a lopsided matchup.
    win_probs = np.array([0.4, 0.5, 0.6])
    spreads = np.array([2.0, 0.0, -2.0])

    isotonic_predictor = IsotonicProbToSpreadFitter().fit(win_probs, spreads)
    logistic_predictor = LogisticProbToSpreadFitter().fit(win_probs, spreads)

    extreme_win_prob = np.array([0.999])
    isotonic_spread = isotonic_predictor.predict_spreads(extreme_win_prob)[0]
    logistic_spread = logistic_predictor.predict_spreads(extreme_win_prob)[0]

    # Isotonic regression has nothing useful to say past the data it saw
    # (sklearn returns nan by default for out-of-range x), while the
    # logistic fit keeps extrapolating a bigger spread.
    assert np.isnan(isotonic_spread) or isotonic_spread == pytest.approx(-2.0)
    assert logistic_spread < -10.0
