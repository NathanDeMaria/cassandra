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
