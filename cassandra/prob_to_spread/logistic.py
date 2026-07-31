import numpy as np

from .base_fit import BaseProbToSpreadFitter, BaseProbToSpreadPredictor

_EPS = 1e-6


def _logit(win_probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(win_probs, _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped))


class LogisticProbToSpreadPredictor(BaseProbToSpreadPredictor):
    def __init__(self, scale: float) -> None:
        self._scale = scale

    def predict_spreads(self, win_probs: np.ndarray) -> np.ndarray:
        return -self._scale * _logit(win_probs)


class LogisticProbToSpreadFitter(BaseProbToSpreadFitter):
    """
    Fits spread = -scale * logit(win_prob).

    This is the natural inverse of Elo's own win-probability formula, which
    is already a logistic function of rating difference. Unlike isotonic
    regression, it's smooth, strictly monotonic, and extrapolates linearly
    for win probabilities outside the observed spread data instead of
    plateauing at the most extreme observed spread.
    """

    def fit(
        self, win_probs: np.ndarray, spreads: np.ndarray
    ) -> BaseProbToSpreadPredictor:
        logits = _logit(win_probs)
        # Least squares through the origin: logit(0.5) = 0 <-> spread = 0,
        # so a pick'em game should map to a pick'em spread.
        scale = -np.dot(logits, spreads) / np.dot(logits, logits)
        return LogisticProbToSpreadPredictor(scale)
