from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from cassandra.columns import GameDfColumns


class BaseProbToMarginPredictor(ABC):
    @abstractmethod
    def predict_margins(self, win_probs: np.ndarray) -> np.ndarray:
        pass


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
