from typing import Any, Self

import numpy as np

from .base_fit import BaseProbToMarginFitter, BaseProbToMarginPredictor


class IsotonicProbToMarginPredictor(BaseProbToMarginPredictor):
    """An isotonic fit stored as its knots and evaluated with np.interp.

    Keeping the knots instead of the fitted estimator is what lets the fit go
    through JSON and come back without scikit-learn, and what keeps a model
    saved last season readable after an sklearn upgrade rewrites its pickles.
    np.interp over (X_thresholds_, y_thresholds_) reproduces
    IsotonicRegression.predict exactly inside the fitted range.
    """

    kind = "isotonic"

    def __init__(self, x_thresholds: np.ndarray, y_thresholds: np.ndarray) -> None:
        self._x_thresholds = np.asarray(x_thresholds, dtype=float)
        self._y_thresholds = np.asarray(y_thresholds, dtype=float)

    def predict_margins(self, win_probs: np.ndarray) -> np.ndarray:
        # np.interp clamps to the end knots outside the fitted range, where
        # IsotonicRegression returns nan (out_of_bounds="nan" is its default).
        # The fit only spans the win probs a season happened to contain --
        # roughly [0.05, 0.95] -- so a lopsided matchup lands outside it, and
        # the nan would surface as a blank number on a page a long way from
        # here. Flattening at the most extreme margin observed is at least an
        # answer; a smooth extrapolation is what LogisticProbToMarginFitter is
        # for.
        return np.interp(win_probs, self._x_thresholds, self._y_thresholds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "x_thresholds": [float(x) for x in self._x_thresholds],
            "y_thresholds": [float(y) for y in self._y_thresholds],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            x_thresholds=np.asarray(data["x_thresholds"], dtype=float),
            y_thresholds=np.asarray(data["y_thresholds"], dtype=float),
        )


class IsotonicProbToMarginFitter(BaseProbToMarginFitter):
    def fit(
        self, win_probs: np.ndarray, margins: np.ndarray
    ) -> BaseProbToMarginPredictor:
        # Imported inside fit, not at module scope, so that a consumer who
        # only rehydrates a saved predictor can import this package with numpy
        # alone and leave scikit-learn out of its image.
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(increasing=True)
        iso.fit(win_probs, margins)
        # X_thresholds_ comes out ascending, which is all np.interp asks of it.
        return IsotonicProbToMarginPredictor(iso.X_thresholds_, iso.y_thresholds_)
