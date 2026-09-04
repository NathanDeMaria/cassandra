import json

import numpy as np
import pytest

from .base_fit import BaseProbToMarginPredictor
from .isotonic import IsotonicProbToMarginFitter
from .logistic import (
    LogisticProbToMarginFitter,
    MaeLogisticProbToMarginFitter,
    _logit,
)


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


def test_all_pick_em_fits_a_flat_zero_instead_of_nan() -> None:
    """The flat baseline calls every game 50/50, so there's no slope to fit.

    Every logit is 0, which zeroes both sides of the least-squares ratio.
    Left alone that's 0/0 -- a nan scale that poisons every prediction and
    drops margin_mae out of the evaluation entirely.
    """
    win_probs = np.full(6, 0.5)
    margins = np.array([21.0, -21.0, 14.0, -8.0, 3.0, -9.0])

    predictor = LogisticProbToMarginFitter().fit(win_probs, margins)

    predicted = predictor.predict_margins(np.linspace(0.05, 0.95, 20))
    assert not np.isnan(predicted).any()
    assert predicted == pytest.approx(0.0)


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


def test_mae_fit_beats_the_least_squares_one_on_absolute_error() -> None:
    """The reason the MAE fitter exists, on the shape of data that produces it.

    Margins are right-skewed: a game can be won by 40 and can't be lost by
    less than 1. The mean margin at a given win probability therefore sits
    above the median, so a least-squares scale predicts wider than the number
    that minimizes absolute error. `publish._best_fit` and the `margin_mae`
    objective both judge on absolute error, so this is the fit they should be
    choosing.
    """
    rng = np.random.default_rng(11)
    win_probs = rng.uniform(0.55, 0.95, 2000)
    logits = _logit(win_probs)
    # Skewed noise: mostly close games, occasionally a blowout.
    margins = 10 * logits + rng.exponential(8, win_probs.size) - 8

    least_squares = LogisticProbToMarginFitter().fit(win_probs, margins)
    mae_fit = MaeLogisticProbToMarginFitter().fit(win_probs, margins)

    def mae(predictor: BaseProbToMarginPredictor) -> float:
        return float(np.abs(predictor.predict_margins(win_probs) - margins).mean())

    assert mae(mae_fit) < mae(least_squares)


def test_mae_fit_recovers_a_known_scale() -> None:
    true_scale = 9.0
    win_probs = np.array([0.55, 0.6, 0.7, 0.8, 0.9, 0.95])
    margins = true_scale * _logit(win_probs)

    predictor = MaeLogisticProbToMarginFitter().fit(win_probs, margins)

    assert predictor.predict_margins(win_probs) == pytest.approx(margins)


def test_mae_fit_ignores_a_blowout_the_least_squares_fit_chases() -> None:
    """A weighted median moves with the middle game, not the extreme one."""
    win_probs = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
    margins = np.array([7.0, 7.0, 7.0, 7.0, 70.0])

    mae_fit = MaeLogisticProbToMarginFitter().fit(win_probs, margins)
    least_squares = LogisticProbToMarginFitter().fit(win_probs, margins)

    assert mae_fit.predict_margins(np.array([0.7])) == pytest.approx([7.0])
    assert least_squares.predict_margins(np.array([0.7]))[0] > 7.0


def test_mae_fit_still_maps_a_pick_em_to_a_level_game() -> None:
    predictor = MaeLogisticProbToMarginFitter().fit(
        win_probs=np.array([0.4, 0.6, 0.8]), margins=np.array([-5.0, 5.0, 12.0])
    )

    assert predictor.predict_margins(np.array([0.5])) == pytest.approx(0.0)


def test_mae_fit_of_an_all_pick_em_model_is_a_flat_zero() -> None:
    """The flat baseline again: every logit is 0, so no slope is identifiable.

    The least-squares fitter answers 0 here, and this one has to agree --
    `publish._best_fit` compares them, and a nan scale would sort ahead of a
    real one rather than out of contention.
    """
    predictor = MaeLogisticProbToMarginFitter().fit(
        win_probs=np.full(4, 0.5), margins=np.array([3.0, -3.0, 10.0, -1.0])
    )

    assert predictor.predict_margins(np.array([0.5, 0.9])) == pytest.approx([0.0, 0.0])


def test_mae_fit_serializes_as_an_ordinary_logistic_calibration() -> None:
    """Same `kind` on the wire, so nothing at serving time has to learn about it."""
    predictor = MaeLogisticProbToMarginFitter().fit(
        win_probs=np.array([0.6, 0.8]), margins=np.array([4.0, 11.0])
    )

    data = json.loads(json.dumps(predictor.to_dict()))
    rehydrated = BaseProbToMarginPredictor.from_dict(data)

    assert data["kind"] == "logistic"
    assert rehydrated.predict_margins(np.array([0.75])) == pytest.approx(
        predictor.predict_margins(np.array([0.75]))
    )
