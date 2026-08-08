import numpy as np
import pandas as pd
import pytest

from .model_eval import score_predictions
from .prob_to_spread import BaseProbToSpreadFitter, BaseProbToSpreadPredictor


class _FixedSpreadFitter(BaseProbToSpreadFitter):
    """Predicts one spread for every game, so the metrics are hand-checkable."""

    def __init__(self, spread: float) -> None:
        self._spread = spread

    def fit(
        self, win_probs: np.ndarray, spreads: np.ndarray
    ) -> BaseProbToSpreadPredictor:
        return _FixedSpreadPredictor(self._spread)


class _FixedSpreadPredictor(BaseProbToSpreadPredictor):
    def __init__(self, spread: float) -> None:
        self._spread = spread

    def predict_spreads(self, win_probs: np.ndarray) -> np.ndarray:
        return np.full(len(win_probs), self._spread)


def _games() -> pd.DataFrame:
    # Game 3 has no line, so it counts toward brier_score and nothing else.
    return pd.DataFrame(
        [
            {
                "home_score": 24,
                "away_score": 20,  # team1_mov = +4
                "team1_win": True,
                "team1_win_prob": 0.7,
                "spread": -3.0,  # team1 favored by 3, and covers
            },
            {
                "home_score": 10,
                "away_score": 30,  # team1_mov = -20
                "team1_win": False,
                "team1_win_prob": 0.4,
                "spread": 6.0,  # team1 getting 6, and does not cover
            },
            {
                "home_score": 14,
                "away_score": 7,
                "team1_win": True,
                "team1_win_prob": 0.6,
                "spread": None,
            },
        ]
    )


def test_margin_mae_reads_the_spread_from_team1s_side() -> None:
    """A model that nails the actual margin has to score zero error.

    The sign is the trap: a spread is quoted from team1's side, so the margin
    it implies is its negation. Flipping it leaves an MAE that still looks
    like a plausible number of points.
    """
    game = _games().iloc[[0]]  # team1 wins by 4

    metrics = score_predictions(game, _FixedSpreadFitter(-4.0))

    assert metrics["margin_mae"] == pytest.approx(0.0)


def test_scores_margin_against_the_market() -> None:
    # Predicted margin +5 against actual +4 and -20: errors of 1 and 25.
    # The market's -3 and +6 imply +3 and -6: errors of 1 and 14.
    metrics = score_predictions(_games(), _FixedSpreadFitter(-5.0))

    assert metrics["margin_mae"] == pytest.approx(13.0)
    assert metrics["market_margin_mae"] == pytest.approx(7.5)


def test_against_spread_accuracy_and_counts() -> None:
    # Predicting -5 beats the -3 line on game 1 (bet team1, and team1 covers)
    # and also beats the +6 line on game 2 (bet team1, but team1 doesn't).
    metrics = score_predictions(_games(), _FixedSpreadFitter(-5.0))

    assert metrics["against_spread_accuracy"] == pytest.approx(0.5)
    assert metrics["n_games"] == 3
    assert metrics["n_spread_games"] == 2


def test_scores_a_league_with_no_lines_at_all() -> None:
    """A league the odds database doesn't cover still gets a brier score.

    The fitters can't be handed an empty frame -- sklearn's isotonic
    regression raises on zero samples -- so the spread metrics drop out.
    """
    no_lines = _games().assign(spread=None)

    metrics = score_predictions(no_lines, _FixedSpreadFitter(-5.0))

    assert metrics["brier_score"] == pytest.approx((0.3**2 + 0.4**2 + 0.4**2) / 3)
    assert metrics["n_games"] == 3
    assert metrics["n_spread_games"] == 0
    assert np.isnan(metrics["margin_mae"])
    assert np.isnan(metrics["against_spread_accuracy"])
    assert np.isnan(metrics["market_margin_mae"])


def test_brier_score_covers_games_without_a_line() -> None:
    metrics = score_predictions(_games(), _FixedSpreadFitter(-5.0))

    expected = (0.3**2 + 0.4**2 + 0.4**2) / 3
    assert metrics["brier_score"] == pytest.approx(expected)
