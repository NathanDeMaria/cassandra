from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from .brier import brier_score_df
from .columns import GameDfColumns
from .predictor import load_predictor
from .prob_to_spread import BaseProbToSpreadFitter, LogisticProbToSpreadFitter
from .save_predictions import build_predictions_df


async def evaluate_model(
    predictor_config_path: Path,
    league: str,
    fitter: BaseProbToSpreadFitter = LogisticProbToSpreadFitter(),
) -> dict[str, float]:
    predictor = load_predictor(predictor_config_path)
    df = await build_predictions_df(predictor, league, post_callbacks=False)
    predictor.save_state(
        predictor_config_path.parent / f"{predictor_config_path.stem}_state.json"
    )

    brier_score = brier_score_df(df)

    with_spread = df[df[GameDfColumns.SPREAD].notna()]
    spread_predictor = fitter.fit_df(with_spread)

    with_spread = with_spread.assign(
        predicted_spread=lambda x: spread_predictor.predict_spreads(
            x[GameDfColumns.TEAM1_WIN_PROB].to_numpy()
        )
    )
    with_spread = with_spread.assign(
        team1_mov=lambda df: df.home_score - df.away_score,
        bet_team1=lambda df: df.predicted_spread < df.spread,
        team1_covered=lambda df: df.spread + df.team1_mov > 0,
        correct_bet=lambda df: df.bet_team1 == df.team1_covered,
    )
    against_spread_accuracy = with_spread["correct_bet"].mean()

    return {
        "brier_score": brier_score,
        "against_spread_accuracy": against_spread_accuracy,
        "n_games": len(df),
        "n_spread_games": len(with_spread) // 2,
    }
