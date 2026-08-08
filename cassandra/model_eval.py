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


async def get_predictions(
    predictor_config_path: Path, league: str, state_path: Path
) -> pd.DataFrame:
    """Run a predictor over a league's games. The expensive, once-per-predictor step.

    `state_path` is explicit because the config can be a checked-in baseline,
    and the state it produces is generated output that doesn't belong there.
    """
    predictor = load_predictor(predictor_config_path)
    df = await build_predictions_df(predictor, league, post_callbacks=False)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_state(state_path)
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
    if with_spread.empty:
        # A league the odds database doesn't cover at all. There's nothing to
        # fit a prob-to-spread model on, but the brier score never needed a
        # line, so report that rather than failing the whole evaluation.
        return {
            "brier_score": brier_score_df(df),
            "against_spread_accuracy": float("nan"),
            "margin_mae": float("nan"),
            "market_margin_mae": float("nan"),
            "n_games": len(df),
            "n_spread_games": 0,
        }

    spread_predictor = fitter.fit_df(with_spread)

    scored = with_spread.assign(
        predicted_spread=lambda x: spread_predictor.predict_spreads(
            x[GameDfColumns.TEAM1_WIN_PROB].to_numpy()
        )
    )
    scored = scored.assign(
        team1_mov=lambda df: df.home_score - df.away_score,
        # A spread is quoted from team1's side and team1 covers when
        # spread + mov > 0, so the margin a line implies is its negation.
        # Getting this backwards still produces a plausible-looking MAE.
        predicted_margin=lambda df: -df.predicted_spread,
        market_margin=lambda df: -df.spread,
        bet_team1=lambda df: df.predicted_spread < df.spread,
        team1_covered=lambda df: df.spread + df.team1_mov > 0,
        correct_bet=lambda df: df.bet_team1 == df.team1_covered,
    )
    return {
        # Scored over every game, unlike the metrics below it, which only have
        # the ones a book put a line on.
        "brier_score": brier_score_df(df),
        "against_spread_accuracy": scored["correct_bet"].mean(),
        "margin_mae": (scored["predicted_margin"] - scored["team1_mov"]).abs().mean(),
        # The same error for the closing line. Margin MAE means little on its
        # own, since game-to-game noise sets a floor no model gets under; the
        # number that carries information is the gap between these two.
        "market_margin_mae": (scored["market_margin"] - scored["team1_mov"])
        .abs()
        .mean(),
        "n_games": len(df),
        "n_spread_games": len(scored),
    }
