from typing import Any, Self

import numpy as np

from .base_fit import BaseProbToMarginFitter, BaseProbToMarginPredictor

_EPS = 1e-6


def _logit(win_probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(win_probs, _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped))


class LogisticProbToMarginPredictor(BaseProbToMarginPredictor):
    kind = "logistic"

    def __init__(self, scale: float) -> None:
        self._scale = scale

    def predict_margins(self, win_probs: np.ndarray) -> np.ndarray:
        return self._scale * _logit(win_probs)

    def to_dict(self) -> dict[str, Any]:
        # float() because the fitted scale arrives as an np.float64.
        return {"kind": self.kind, "scale": float(self._scale)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(scale=float(data["scale"]))


class LogisticProbToMarginFitter(BaseProbToMarginFitter):
    """
    Fits margin = scale * logit(win_prob).

    This is the natural inverse of Elo's own win-probability formula, which
    is already a logistic function of rating difference. Unlike isotonic
    regression, it's smooth, strictly monotonic, and extrapolates linearly
    for win probabilities outside the observed range instead of plateauing
    at the most extreme margin it happened to see.
    """

    def fit(
        self, win_probs: np.ndarray, margins: np.ndarray
    ) -> BaseProbToMarginPredictor:
        logits = _logit(win_probs)
        # Least squares through the origin: logit(0.5) = 0 <-> margin = 0,
        # so a pick'em game should be predicted to finish level.
        denominator = np.dot(logits, logits)
        if denominator == 0:
            # A predictor that calls every game a coin flip (flat) has every
            # logit at 0, so there's no slope to fit. Predict level games
            # rather than propagating a nan scale into every margin.
            return LogisticProbToMarginPredictor(0.0)
        scale = np.dot(logits, margins) / denominator
        return LogisticProbToMarginPredictor(scale)


class MaeLogisticProbToMarginFitter(BaseProbToMarginFitter):
    """The same margin = scale * logit(win_prob), fit to minimize MAE.

    `LogisticProbToMarginFitter` fits by least squares, so its scale tracks
    the *mean* margin at a given win probability. Mean absolute error is
    minimized by the *median* instead, and the two differ here: a blowout
    pulls the mean and leaves the median alone, so a least-squares scale
    predicts margins a couple of points wider than the number that would
    have scored best on `margin_mae`.

    That matters because `margin_mae` is what `publish._best_fit` chooses on
    and what the `margin_mae` optimization objective searches against. A
    model scored through a fit that is systematically too wide is being
    charged for the fit's loss function rather than its own ratings.

    Emits a `LogisticProbToMarginPredictor` -- the same `kind` on the wire,
    the same arithmetic at serving time. Only the number differs, so nothing
    downstream has to learn about this.
    """

    def fit(
        self, win_probs: np.ndarray, margins: np.ndarray
    ) -> BaseProbToMarginPredictor:
        logits = _logit(win_probs)
        # Through the origin, like the least-squares fit above, so a pick'em
        # game still predicts a level one:
        #   sum |scale * logit - margin| = sum |logit| * |scale - margin/logit|
        # which is a weighted median of the per-game slopes, weighted by how
        # far from a coin flip each game was called.
        nonzero = logits != 0
        slopes = margins[nonzero] / logits[nonzero]
        weights = np.abs(logits[nonzero])
        if weights.size == 0 or weights.sum() == 0:
            # Every game called a coin flip (flat), so no slope is
            # identifiable. Predict level games rather than dividing by zero
            # -- the same answer the least-squares fitter gives.
            return LogisticProbToMarginPredictor(0.0)
        order = np.argsort(slopes)
        sorted_slopes = slopes[order]
        cumulative = np.cumsum(weights[order])
        # The first slope at which half the weight is behind us. searchsorted
        # with "left" lands on the lower of the two medians when the weight
        # splits exactly; any point between them scores identically, so the
        # choice is arbitrary rather than approximate.
        index = int(np.searchsorted(cumulative, cumulative[-1] / 2))
        scale = sorted_slopes[min(index, sorted_slopes.size - 1)]
        return LogisticProbToMarginPredictor(float(scale))
