from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from cassandra.columns import GameDfColumns

_PREDICTOR_KINDS: dict[str, type["BaseProbToMarginPredictor"]] = {}


class BaseProbToMarginPredictor(ABC):
    """A fitted prob-to-margin mapping, separable from whatever fit it.

    Predictors are serializable on purpose. The fitting happens here, but the
    thing that asks "what margin does a 62% win probability imply?" is a web
    service that has numpy and not much else. So `to_dict` emits plain JSON
    types, and `from_dict` rebuilds a working predictor without touching a
    fitter or the library that produced it.
    """

    #: Tag written into `to_dict` output and used to dispatch `from_dict`.
    #: Setting it on a subclass registers that subclass.
    kind: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # __dict__ rather than getattr: only a subclass that declares its own
        # kind registers, so an intermediate subclass can't hijack the name.
        kind = cls.__dict__.get("kind")
        if kind is not None:
            _PREDICTOR_KINDS[kind] = cls

    @abstractmethod
    def predict_margins(self, win_probs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """The fit as plain JSON-safe data, tagged with `kind`.

        Values have to be builtins, not numpy scalars: np.float64 only
        survives json.dumps by accident of subclassing float, and anything
        else in the numpy float family raises.
        """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseProbToMarginPredictor":
        """Rebuild the predictor `data` describes, dispatching on its kind."""
        kind = data["kind"]
        subclass = _PREDICTOR_KINDS.get(kind)
        if subclass is None:
            raise ValueError(f"unknown prob-to-margin predictor kind: {kind!r}")
        if subclass is cls:
            raise NotImplementedError(f"{cls.__name__} must override from_dict")
        return subclass.from_dict(data)


class BaseProbToMarginFitter(ABC):
    @abstractmethod
    def fit(
        self, win_probs: np.ndarray, margins: np.ndarray
    ) -> BaseProbToMarginPredictor:
        pass

    def fit_df(self, games: pd.DataFrame) -> BaseProbToMarginPredictor:
        """
        Fit the model using a dataframe with team1_win_prob and team1_mov columns.

        The target is the margin the game actually finished at, not the line a
        book hung on it, so every game with a final score is training data --
        many times what the subset with a closing spread gives us.

        This function will flip the dataframe and fit the model on both the
        original and flipped data for the 'away team' side.
        """
        flipped = pd.concat(
            [
                games,
                games.assign(
                    team1_mov=-games[GameDfColumns.TEAM1_MOV],
                    team1_win_prob=1 - games[GameDfColumns.TEAM1_WIN_PROB],
                ),
            ],
            ignore_index=True,
        )
        return self.fit(
            win_probs=flipped[GameDfColumns.TEAM1_WIN_PROB].to_numpy(),
            margins=flipped[GameDfColumns.TEAM1_MOV].to_numpy(),
        )
