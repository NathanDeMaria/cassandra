import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from .base_fit import BaseProbToMarginPredictor
from .kernel import (
    MarginKernel,
    MarginKernelFitter,
    _negative_log_likelihood,
    _start,
)
from .logistic import _logit


def _football(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Games whose margins land on 3 and 7 far more than their neighbours.

    A margin of 12 points per logit with logistic noise of the same scale,
    so a game called at p is won with probability p, and then a fifth of the
    games snapped to the nearest of +-3 and +-7 -- the lumps the fit should
    find without being told where they are.
    """
    rng = np.random.default_rng(seed)
    win_probs = rng.uniform(0.15, 0.85, n)
    margins = np.rint(12 * _logit(win_probs) + rng.logistic(0, 12, n))
    keys = np.array([-7, -3, 3, 7])
    snapped = rng.random(n) < 0.2
    nearest = keys[np.abs(margins[:, None] - keys[None, :]).argmin(axis=1)]
    return win_probs, np.where(snapped, nearest, margins)


@pytest.fixture(scope="module")
def kernel() -> MarginKernel:
    win_probs, margins = _football()
    return MarginKernelFitter(max_margin=40).fit(win_probs, margins)


def test_the_gradient_is_the_likelihoods() -> None:
    rng = np.random.default_rng(1)
    k = 6
    eta = rng.normal(0, 1, 300)
    cells = np.clip(np.rint(5 * eta + rng.normal(0, 4, 300)).astype(int), -k, k) + k
    params = _start(cells, k) + rng.normal(0, 0.2, k + 1)

    _, analytic = _negative_log_likelihood(params, eta, cells, k)
    numeric = approx_fprime(
        params, lambda x: _negative_log_likelihood(x, eta, cells, k)[0], 1e-6
    )
    assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-4)


def test_it_finds_the_key_numbers(kernel: MarginKernel) -> None:
    pmf = kernel.distribution(0.5).pmf
    zero = kernel.max_margin
    for key in (3, 7):
        assert pmf[zero + key] > 1.5 * max(pmf[zero + key - 1], pmf[zero + key + 1])
        assert pmf[zero - key] > 1.5 * max(pmf[zero - key - 1], pmf[zero - key + 1])


def test_every_row_is_a_distribution(kernel: MarginKernel) -> None:
    pmfs = kernel.pmfs(np.array([0.01, 0.3, 0.5, 0.9, 0.999]))
    assert (pmfs >= 0).all()
    assert pmfs.sum(axis=1) == pytest.approx(1.0)
    assert pmfs.shape == (5, 2 * kernel.max_margin + 1)


def test_a_game_from_the_other_side_is_its_mirror_image(kernel: MarginKernel) -> None:
    home = kernel.pmfs(np.array([0.7]))[0]
    away = kernel.pmfs(np.array([0.3]))[0]
    assert home == pytest.approx(away[::-1])


def test_a_likelier_win_is_a_bigger_margin(kernel: MarginKernel) -> None:
    means = [kernel.distribution(p).mean for p in (0.3, 0.5, 0.7, 0.9)]
    assert means == sorted(means)
    assert kernel.distribution(0.5).mean == pytest.approx(0.0, abs=1e-9)


def test_its_point_margin_is_the_median(kernel: MarginKernel) -> None:
    for p in (0.25, 0.8):
        median = kernel.predict_margins(np.array([p]))[0]
        distribution = kernel.distribution(p)
        assert distribution.cdf(median) >= 0.5
        assert distribution.cdf(median - 1) < 0.5


def test_its_win_probability_is_near_the_models(kernel: MarginKernel) -> None:
    """Not equal -- the model's stays the one the pipeline reads -- but close."""
    for p in (0.3, 0.6, 0.8):
        assert kernel.distribution(p).win_prob() == pytest.approx(p, abs=0.03)


def test_it_round_trips_through_plain_json(kernel: MarginKernel) -> None:
    data = json.loads(json.dumps(kernel.to_dict()))
    restored = BaseProbToMarginPredictor.from_dict(data)

    assert isinstance(restored, MarginKernel)
    probs = np.array([0.2, 0.5, 0.75])
    assert restored.pmfs(probs) == pytest.approx(kernel.pmfs(probs))


def test_applying_a_saved_kernel_never_imports_scipy(kernel: MarginKernel) -> None:
    """What a web service holding a saved kernel needs: numpy, and no fit group.

    A subprocess, because scipy is in sys.modules by the time this file has
    fit anything. See `release_test.test_reading_a_release_never_imports_sklearn`.
    """
    source = """
import json, sys
from cassandra.prob_to_margin import BaseProbToMarginPredictor

kernel = BaseProbToMarginPredictor.from_dict(json.load(sys.stdin))
assert kernel.distribution(0.7).p_cover(-3.5).cover > 0
print(any(m == "scipy" or m.startswith(("scipy.", "sklearn")) for m in sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        input=json.dumps(kernel.to_dict()),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"


def test_margins_past_the_range_land_in_the_end_cells() -> None:
    win_probs, margins = _football(n=2_000)
    margins[:50] = 90
    kernel = MarginKernelFitter(max_margin=30).fit(win_probs, margins)
    assert kernel.distribution(0.99).pmf[-1] > 0.01


def test_a_model_that_calls_every_game_a_coin_flip_still_fits() -> None:
    """flat: no slope to find, but the margin's own shape is still there."""
    _, margins = _football(n=2_000)
    kernel = MarginKernelFitter(max_margin=40).fit(np.full(2_000, 0.5), margins)

    pmf = kernel.distribution(0.5).pmf
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf == pytest.approx(pmf[::-1])


def test_max_margin_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_margin"):
        MarginKernelFitter(max_margin=0)
