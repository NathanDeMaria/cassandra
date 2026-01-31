import numpy as np
from sklearn.isotonic import IsotonicRegression

from .base_fit import BaseProbToSpreadFitter, BaseProbToSpreadPredictor


class IsotonicProbToSpreadPredictor(BaseProbToSpreadPredictor):
    def __init__(self, iso: IsotonicRegression) -> None:
        self._iso = iso

    def predict_spreads(self, win_probs: np.ndarray) -> np.ndarray:
        return self._iso.predict(win_probs)


class IsotonicProbToSpreadFitter(BaseProbToSpreadFitter):
    def fit(
        self, win_probs: np.ndarray, spreads: np.ndarray
    ) -> BaseProbToSpreadPredictor:
        iso = IsotonicRegression(increasing=False)
        iso.fit(win_probs, spreads)
        return IsotonicProbToSpreadPredictor(iso)
