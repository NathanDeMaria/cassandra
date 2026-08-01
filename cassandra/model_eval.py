from pathlib import Path

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


async def get_predictions(predictor_config_path: Path, league: str) -> pd.DataFrame:
    """Run a predictor over a league's games. The expensive, once-per-predictor step."""
    predictor = load_predictor(predictor_config_path)
    df = await build_predictions_df(predictor, league, post_callbacks=False)
    predictor.save_state(
        predictor_config_path.parent / f"{predictor_config_path.stem}_state.json"
    )
    return df


def score_predictions(
    df: pd.DataFrame, fitter: BaseProbToSpreadFitter
) -> dict[str, float]:
    """Score a predictor's predictions against a single prob-to-spread fitter.

    Cheap relative to get_predictions, so callers comparing multiple fitters
    should call this once per fitter on the same df rather than re-fetching
    predictions each time.
    """
    with_spread = df[df[GameDfColumns.SPREAD].notna()]
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
    return {
        "brier_score": brier_score_df(df),
        "against_spread_accuracy": scored["correct_bet"].mean(),
        "n_games": len(df),
        "n_spread_games": len(scored) // 2,
    }
