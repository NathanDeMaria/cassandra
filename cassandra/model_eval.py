from pathlib import Path
from typing import NamedTuple

import pandas as pd

from .brier import brier_score_df
from .columns import GameDfColumns
from .predictor import load_predictor
from .prob_to_margin import (
    BaseProbToMarginFitter,
    BaseProbToMarginPredictor,
    IsotonicProbToMarginFitter,
    LogisticProbToMarginFitter,
    MaeLogisticProbToMarginFitter,
)
from .save_predictions import build_predictions_df

DEFAULT_FITTERS: dict[str, BaseProbToMarginFitter] = {
    "isotonic": IsotonicProbToMarginFitter(),
    "logistic": LogisticProbToMarginFitter(),
    # The least-squares logistic's sibling, fit on the loss `_best_fit` and
    # the `margin_mae` objective actually judge by. Added rather than
    # replacing "logistic": the two answer different questions -- expected
    # margin vs. the margin to bet -- and a release that wants the mean can
    # still be pointed at the fit that estimates it.
    "logistic_mae": MaeLogisticProbToMarginFitter(),
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


class ScoredPredictions(NamedTuple):
    """Metrics, plus the fit they were computed with.

    The fit used to be discarded once against_spread_accuracy was out of it,
    which left no way to ask what margin a given win probability implies. It
    comes back out so callers can serialize it next to the model's ratings.
    """

    metrics: dict[str, float]
    margin_predictor: BaseProbToMarginPredictor


def score_predictions(
    df: pd.DataFrame, fitter: BaseProbToMarginFitter
) -> ScoredPredictions:
    """Score a predictor's predictions against a single prob-to-margin fitter.

    Cheap relative to get_predictions, so callers comparing multiple fitters
    should call this once per fitter on the same df rather than re-fetching
    predictions each time.
    """
    if df.empty:
        # The same guard `brier_score_df` has, and it has to be here too:
        # that one is called further down, and the assign below reaches for
        # columns an empty frame doesn't have. Without this a caller gets
        # `AttributeError: 'DataFrame' object has no attribute 'home_score'`,
        # which reads like a schema bug rather than "there were no games" --
        # publish spent a run failing that way on a league whose season had
        # been uploaded with nothing in it.
        raise ValueError("No games to score")
    games = df.assign(team1_mov=lambda x: x.home_score - x.away_score)
    # Fit against the margin every game actually finished at, so the fitter
    # trains on the whole schedule instead of only the games a book put a
    # line on. The betting metrics below still need the line, and still only
    # cover that subset.
    margin_predictor = fitter.fit_df(games)
    scored = games.assign(
        predicted_margin=lambda x: margin_predictor.predict_margins(
            x[GameDfColumns.TEAM1_WIN_PROB].to_numpy()
        )
    )
    metrics = {
        "brier_score": brier_score_df(df),
        "margin_mae": (scored["predicted_margin"] - scored["team1_mov"]).abs().mean(),
        "n_games": len(scored),
    }

    with_spread = scored[scored[GameDfColumns.SPREAD].notna()]
    if with_spread.empty:
        # A league the odds database doesn't cover at all. Everything above
        # still holds -- the margin fit never needed a line -- but there's
        # nothing to bet into or compare against.
        return ScoredPredictions(
            metrics={
                **metrics,
                "against_spread_accuracy": float("nan"),
                "spread_game_margin_mae": float("nan"),
                "market_margin_mae": float("nan"),
                "n_spread_games": 0,
            },
            margin_predictor=margin_predictor,
        )

    with_spread = with_spread.assign(
        # A spread is quoted from team1's side and team1 covers when
        # spread + mov > 0, so the margin a line implies is its negation.
        # Getting this backwards still produces a plausible-looking MAE.
        market_margin=lambda x: -x.spread,
        bet_team1=lambda x: x.predicted_margin > x.market_margin,
        team1_covered=lambda x: x.spread + x.team1_mov > 0,
        correct_bet=lambda x: x.bet_team1 == x.team1_covered,
    )
    return ScoredPredictions(
        metrics={
            **metrics,
            "against_spread_accuracy": with_spread["correct_bet"].mean(),
            # margin_mae over the lined games only, which is the slice
            # market_margin_mae covers. Margin MAE means little on its own,
            # since game-to-game noise sets a floor no model gets under; the
            # number that carries information is the gap between these two.
            "spread_game_margin_mae": (
                with_spread["predicted_margin"] - with_spread["team1_mov"]
            )
            .abs()
            .mean(),
            "market_margin_mae": (
                with_spread["market_margin"] - with_spread["team1_mov"]
            )
            .abs()
            .mean(),
            "n_spread_games": len(with_spread),
        },
        margin_predictor=margin_predictor,
    )
