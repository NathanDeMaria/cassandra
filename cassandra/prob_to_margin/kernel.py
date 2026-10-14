"""A win probability as a probability on every integer margin.

`LogisticProbToMarginPredictor` answers "what margin does 62% imply?" with
one number. A spread asks something else -- how likely the home side is to
win by more than 7, exactly 7, or less -- and the answer depends on the
shape of the margin, not its centre. Football's shape is lumpy: 3 and 7 are
where games end far more than their neighbours, so a smooth curve through
the mean misprices exactly the lines that matter.

The model
---------

A cumulative logit on the margin, with the win probability's logit as the
one covariate:

    P(margin <= k) = sigmoid(c_k - slope * logit(p))

with a free cutpoint `c_k` between every pair of adjacent integers. Nothing
says where the key numbers are; a cutpoint pair far apart is a margin games
often end on, and the fit finds 3 and 7 (and 10, 14, 21, 24) on its own --
that was the scratch study that led here (`clm.py`, 2026-09-26), where it
beat both a normal and a hand-keyed normal on cover brier.

The cutpoints are **symmetric**, `c_k = -c_{-k-1}`, so a game called at p
from one side is the same game called at 1 - p from the other. That halves
the parameters, and makes the flipped copy of the games that
`BaseProbToMarginFitter.fit_df` adds a no-op: a game and its mirror image
already have the same likelihood.

Margins past `max_margin` are folded into the end cells.

As a prob->margin fit
---------------------

It is one -- `predict_margins` is the median, the margin MAE is minimized
by -- so `model_eval.score_predictions` scores it like the others and a
release can carry it later through the same `from_dict` dispatch. It is
*not* in `DEFAULT_FITTERS`, so it never competes for the calibration a
release ships with.

What it is not
--------------

Its P(margin > 0) is not the model's win probability; close, and the gap is
reported (`kernel_win_prob_gap`), but the model's own number stays the one
the pipeline reads. The kernel is for spreads and margins only.

Like the calibrations beside it, applying a fit needs numpy and nothing
else, and `to_dict` is plain JSON. Fitting needs scipy, imported inside
`fit` for the same reason `IsotonicProbToMarginFitter` imports sklearn there.
"""

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .base_fit import BaseProbToMarginFitter, BaseProbToMarginPredictor
from .logistic import _logit

if TYPE_CHECKING:
    from cassandra.predictor.distribution import DiscreteMargin

DEFAULT_MAX_MARGIN = 60

# Bounds on the log of each cutpoint gap. The bottom is a margin no game in
# the history ended on -- its cell is empty, and the fit would otherwise
# chase its gap to zero forever -- and the top is far beyond any gap a fit
# wants, there only so a bad start can't overflow.
_LOG_GAP_BOUNDS = (-15.0, 5.0)
_SLOPE_BOUNDS = (1e-6, 100.0)
_MIN_PROB = 1e-300


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # tanh rather than 1 / (1 + exp(-x)): the same function, without the
    # overflow warning for a cutpoint far out in an empty tail.
    return 0.5 * (1 + np.tanh(x / 2))


def _cutpoints(log_gaps: np.ndarray) -> np.ndarray:
    """Every cutpoint, c_{-K} .. c_{K-1}, from the gaps of the upper half.

    `c_k` for k >= 0 is the running sum of the gaps, and the lower half is
    its mirror image: `c_{-k-1} = -c_k`.
    """
    upper = np.cumsum(np.exp(log_gaps))
    return np.concatenate([-upper[::-1], upper])


class MarginKernel(BaseProbToMarginPredictor):
    """A fitted cumulative logit: win probabilities in, margin pmfs out."""

    kind = "cumulative_logit"

    def __init__(self, slope: float, log_gaps: np.ndarray) -> None:
        self._slope = slope
        self._log_gaps = np.asarray(log_gaps, dtype=float)

    @property
    def max_margin(self) -> int:
        return len(self._log_gaps)

    @property
    def slope(self) -> float:
        return self._slope

    @property
    def cutpoints(self) -> np.ndarray:
        """c_{-K} .. c_{K-1}: P(margin <= k) at a coin flip is sigmoid(c_k)."""
        return _cutpoints(self._log_gaps)

    def cdfs(self, win_probs: np.ndarray) -> np.ndarray:
        """P(margin <= k) for k = -K .. K, one row per game; the last is 1."""
        eta = self._slope * _logit(np.asarray(win_probs, dtype=float))
        below = _sigmoid(self.cutpoints[None, :] - eta[:, None])
        return np.concatenate([below, np.ones((len(eta), 1))], axis=1)

    def pmfs(self, win_probs: np.ndarray) -> np.ndarray:
        """P(margin = k) for k = -K .. K, one row per game."""
        return np.diff(self.cdfs(win_probs), axis=1, prepend=0.0)

    def predict_margins(self, win_probs: np.ndarray) -> np.ndarray:
        """The median margin: the first at which the cdf reaches a half."""
        below_half = self.cdfs(win_probs) < 0.5
        return below_half.sum(axis=1).astype(float) - self.max_margin

    def distribution(self, win_prob: float) -> "DiscreteMargin":
        """One game's margin, as a distribution to read lines against."""
        # Here rather than at the top: the predictor package imports the
        # objectives, which import this package, so a module-level import
        # would be circular.
        from cassandra.predictor.distribution import DiscreteMargin

        return DiscreteMargin(-self.max_margin, self.pmfs(np.array([win_prob]))[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "slope": float(self._slope),
            "log_gaps": [float(g) for g in self._log_gaps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if data["kind"] != cls.kind:
            raise ValueError(f"not a margin kernel: {data['kind']!r}")
        return cls(slope=float(data["slope"]), log_gaps=np.array(data["log_gaps"]))


class MarginKernelFitter(BaseProbToMarginFitter):
    """Fits a `MarginKernel` by maximum likelihood on final margins."""

    def __init__(self, max_margin: int = DEFAULT_MAX_MARGIN) -> None:
        if max_margin < 1:
            raise ValueError(f"max_margin must be at least 1, got {max_margin}")
        self._max_margin = max_margin

    def fit(self, win_probs: np.ndarray, margins: np.ndarray) -> MarginKernel:
        from scipy.optimize import minimize

        k = self._max_margin
        eta = _logit(np.asarray(win_probs, dtype=float))
        cells = np.clip(np.rint(margins).astype(int), -k, k) + k
        result = minimize(
            _negative_log_likelihood,
            _start(cells, k),
            args=(eta, cells, k),
            jac=True,
            method="L-BFGS-B",
            bounds=[_LOG_GAP_BOUNDS] * k + [_SLOPE_BOUNDS],
        )
        return MarginKernel(slope=float(result.x[-1]), log_gaps=result.x[:-1])



def _start(cells: np.ndarray, k: int) -> np.ndarray:
    """The pooled margin's own cutpoints, at a slope of 1.

    Pooled with its mirror image, which is what the symmetric cutpoints can
    represent: the logit of the share of games at or below each margin,
    turned into gaps. A gap the data leaves flat starts small rather than at
    the bottom bound, where its gradient would be nothing.
    """
    counts = np.bincount(cells, minlength=2 * k + 1).astype(float)
    counts = counts + counts[::-1] + 1.0
    share = np.cumsum(counts)[:-1] / counts.sum()
    upper = np.log(share / (1 - share))[k:]
    gaps = np.diff(upper, prepend=0.0)
    log_gaps = np.clip(np.log(np.maximum(gaps, 1e-3)), *_LOG_GAP_BOUNDS)
    return np.append(log_gaps, 1.0)


def _negative_log_likelihood(
    params: np.ndarray, eta: np.ndarray, cells: np.ndarray, k: int
) -> tuple[float, np.ndarray]:
    """The kernel's negative log likelihood, and its gradient in `params`.

    `params` is the log gaps of the upper half, then the slope. A game in
    cell m (margin m - k) has probability F(c_m - b eta) - F(c_{m-1} - b eta),
    with the cutpoints past either end at plus and minus infinity.
    """
    log_gaps, slope = params[:-1], params[-1]
    cutpoints = _cutpoints(log_gaps)
    padded = np.concatenate([[-np.inf], cutpoints, [np.inf]])
    upper = padded[cells + 1] - slope * eta
    lower = padded[cells] - slope * eta
    f_upper, f_lower = _sigmoid(upper), _sigmoid(lower)
    prob = np.maximum(f_upper - f_lower, _MIN_PROB)
    nll = -np.log(prob).sum()

    # dF/dx = F (1 - F), which is 0 at either infinity, as it should be.
    d_upper = f_upper * (1 - f_upper) / prob
    d_lower = f_lower * (1 - f_lower) / prob
    # Into the padded cutpoints, then down to the 2K real ones.
    grad_padded = np.bincount(
        cells + 1, weights=-d_upper, minlength=2 * k + 2
    ) + np.bincount(cells, weights=d_lower, minlength=2 * k + 2)
    grad_cut = grad_padded[1:-1]
    # c_{j-1} = S_j for the upper half and c_{-j} = -S_j for the lower, with
    # S_j the running sum of the first j gaps.
    grad_sums = grad_cut[k:] - grad_cut[:k][::-1]
    grad_log_gaps = np.exp(log_gaps) * np.cumsum(grad_sums[::-1])[::-1]
    grad_slope = float((eta * (d_upper - d_lower)).sum())
    return float(nll), np.append(grad_log_gaps, grad_slope)
