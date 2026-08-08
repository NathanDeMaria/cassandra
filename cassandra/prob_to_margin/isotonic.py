import numpy as np
from sklearn.isotonic import IsotonicRegression

from .base_fit import BaseProbToMarginFitter, BaseProbToMarginPredictor


class IsotonicProbToMarginPredictor(BaseProbToMarginPredictor):
    def __init__(self, iso: IsotonicRegression) -> None:
        self._iso = iso

    def predict_margins(self, win_probs: np.ndarray) -> np.ndarray:
        return self._iso.predict(win_probs)


class IsotonicProbToMarginFitter(BaseProbToMarginFitter):
    def fit(
        self, win_probs: np.ndarray, margins: np.ndarray
    ) -> BaseProbToMarginPredictor:
        iso = IsotonicRegression(increasing=True)
        iso.fit(win_probs, margins)
        return IsotonicProbToMarginPredictor(iso)
