from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from .brier import brier_score_df
from .columns import GameDfColumns
from .predictor import load_predictor
from .prob_to_spread import (
    BaseProbToSpreadFitter,
    IsotonicProbToSpreadFitter,
    LogisticProbToSpreadFitter,
)
from .save_predictions import build_predictions_df

DEFAULT_FITTERS: dict[str, BaseProbToSpreadFitter] = {
    "isotonic": IsotonicProbToSpreadFitter(),
    "logistic": LogisticProbToSpreadFitter(),
}


async def evaluate_model(
    predictor_config_path: Path,
    league: str,
    fitters: dict[str, BaseProbToSpreadFitter] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate a predictor, scoring it against every given prob-to-spread fitter.

    Predictions are only pulled/generated once and reused across fitters,
    since the fitter choice doesn't affect the predictor's own win-prob output.
    """
    fitters = fitters if fitters is not None else DEFAULT_FITTERS

    predictor = load_predictor(predictor_config_path)
    df = await build_predictions_df(predictor, league, post_callbacks=False)
    predictor.save_state(
        predictor_config_path.parent / f"{predictor_config_path.stem}_state.json"
    )

    brier_score = brier_score_df(df)
    with_spread = df[df[GameDfColumns.SPREAD].notna()]

    results: dict[str, dict[str, float]] = {}
    for fitter_name, fitter in fitters.items():
        spread_predictor = fitter.fit_df(with_spread)
        scored = with_spread.assign(
            predicted_spread=lambda x: spread_predictor.predict_spreads(
                x[GameDfColumns.TEAM1_WIN_PROB].to_numpy()
            )
        )
        scored = scored.assign(
            team1_mov=lambda df: df.home_score - df.away_score,
            bet_team1=lambda df: df.predicted_spread < df.spread,
            team1_covered=lambda df: df.spread + df.team1_mov > 0,
            correct_bet=lambda df: df.bet_team1 == df.team1_covered,
        )
        results[fitter_name] = {
            "brier_score": brier_score,
            "against_spread_accuracy": scored["correct_bet"].mean(),
            "n_games": len(df),
            "n_spread_games": len(scored) // 2,
        }

    return results
