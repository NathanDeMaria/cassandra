from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from cassandra.columns import GameDfColumns


class BaseProbToSpreadPredictor(ABC):
    @abstractmethod
    def predict_spreads(self, win_probs: np.ndarray) -> np.ndarray:
        pass


class BaseProbToSpreadFitter(ABC):
    @abstractmethod
    def fit(
        self, win_probs: np.ndarray, spreads: np.ndarray
    ) -> BaseProbToSpreadPredictor:
        pass

    def fit_df(self, with_spread: pd.DataFrame) -> BaseProbToSpreadPredictor:
        """
        Fit the model using a dataframe with team1_win_prob and spread columns.

        This function will flip the dataframe and fit the model on both the original
        and flipped data for the 'away team' side.
        """
        flipped = pd.concat(
            [
                with_spread,
                with_spread.assign(
                    spread=-with_spread[GameDfColumns.SPREAD],
                    team1_win_prob=1 - with_spread[GameDfColumns.TEAM1_WIN_PROB],
                ),
            ],
            ignore_index=True,
        )
        return self.fit(
            win_probs=flipped[GameDfColumns.TEAM1_WIN_PROB].to_numpy(),
            spreads=flipped[GameDfColumns.SPREAD].to_numpy(),
        )
