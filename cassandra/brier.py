import numpy as np
import numpy.typing as npt
import pandas as pd

from .columns import GameDfColumns


def brier_score_df(df: pd.DataFrame) -> float:
    return brier_score(
        df[GameDfColumns.TEAM1_WIN_PROB].to_numpy(),
        df[GameDfColumns.TEAM1_WIN].to_numpy(),
    )


def brier_score(
    team1_win_probs: npt.NDArray[np.float64], team1_wins: npt.NDArray[np.bool_]
) -> float:
    return np.power(team1_win_probs - team1_wins, 2).mean().item()
