from typing import Any

import numpy as np
import pandas as pd
import pytest

from .model_eval import score_predictions
from .prob_to_margin import BaseProbToMarginFitter, BaseProbToMarginPredictor


class _FixedMarginFitter(BaseProbToMarginFitter):
    """Predicts one margin for every game, so the metrics are hand-checkable."""

    def __init__(self, margin: float) -> None:
        self._margin = margin
        self.fit_win_probs: np.ndarray | None = None

    def fit(
        self, win_probs: np.ndarray, margins: np.ndarray
    ) -> BaseProbToMarginPredictor:
        self.fit_win_probs = win_probs
        return _FixedMarginPredictor(self._margin)


class _FixedMarginPredictor(BaseProbToMarginPredictor):
    # No `kind`, so this stays out of the from_dict registry -- a test double
    # has no business being rehydratable by name.
    def __init__(self, margin: float) -> None:
        self._margin = margin

    def predict_margins(self, win_probs: np.ndarray) -> np.ndarray:
        return np.full(len(win_probs), self._margin)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "fixed", "margin": self._margin}


def _games() -> pd.DataFrame:
    # Game 3 has no line, so it's still training data and still counts toward
    # brier_score and margin_mae, but not toward any of the betting metrics.
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
                "away_score": 7,  # team1_mov = +7
                "team1_win": True,
                "team1_win_prob": 0.6,
                "spread": None,
            },
        ]
    )


def test_fits_on_every_game_not_just_the_lined_ones() -> None:
    """The whole point of predicting margin instead of the spread.

    Three games, one of which no book put a line on, all reach the fitter --
    six rows once fit_df mirrors them onto the away side.
    """
    fitter = _FixedMarginFitter(0.0)

    score_predictions(_games(), fitter)

    assert fitter.fit_win_probs is not None
    assert sorted(fitter.fit_win_probs) == pytest.approx([0.3, 0.4, 0.4, 0.6, 0.6, 0.7])


def test_margin_mae_is_zero_for_a_perfect_prediction() -> None:
    game = _games().iloc[[0]]  # team1 wins by 4

    metrics = score_predictions(game, _FixedMarginFitter(4.0)).metrics

    assert metrics["margin_mae"] == pytest.approx(0.0)


def test_scores_margin_against_the_market() -> None:
    """The market's side of the comparison is where the sign is a trap.

    A spread is quoted from team1's side, so the margin it implies is its
    negation. Flipping it leaves an MAE that still looks like a plausible
    number of points.
    """
    # Predicted margin +5 against actual +4, -20 and +7: errors of 1, 25, 2.
    # The market's -3 and +6 imply +3 and -6: errors of 1 and 14.
    metrics = score_predictions(_games(), _FixedMarginFitter(5.0)).metrics

    assert metrics["margin_mae"] == pytest.approx(28 / 3)
    assert metrics["spread_game_margin_mae"] == pytest.approx(13.0)
    assert metrics["market_margin_mae"] == pytest.approx(7.5)


def test_against_spread_accuracy_and_counts() -> None:
    # Predicting +5 beats the -3 line on game 1 (bet team1, and team1 covers)
    # and also beats the +6 line on game 2 (bet team1, but team1 doesn't).
    metrics = score_predictions(_games(), _FixedMarginFitter(5.0)).metrics

    assert metrics["against_spread_accuracy"] == pytest.approx(0.5)
    assert metrics["n_games"] == 3
    assert metrics["n_spread_games"] == 2


def test_scores_a_league_with_no_lines_at_all() -> None:
    """A league the odds database doesn't cover still gets margin metrics.

    Fitting on final scores rather than on closing lines means these leagues
    are no longer a special case -- only the metrics that need a line to
    compare against drop out.
    """
    no_lines = _games().assign(spread=None)

    metrics = score_predictions(no_lines, _FixedMarginFitter(5.0)).metrics

    assert metrics["brier_score"] == pytest.approx((0.3**2 + 0.4**2 + 0.4**2) / 3)
    assert metrics["margin_mae"] == pytest.approx(28 / 3)
    assert metrics["n_games"] == 3
    assert metrics["n_spread_games"] == 0
    assert np.isnan(metrics["spread_game_margin_mae"])
    assert np.isnan(metrics["against_spread_accuracy"])
    assert np.isnan(metrics["market_margin_mae"])


def test_brier_score_covers_games_without_a_line() -> None:
    metrics = score_predictions(_games(), _FixedMarginFitter(5.0)).metrics

    expected = (0.3**2 + 0.4**2 + 0.4**2) / 3
    assert metrics["brier_score"] == pytest.approx(expected)


def test_a_league_with_no_games_says_so() -> None:
    """
    A season can exist and hold nothing -- an upload made before the league
    started playing. That used to surface as an AttributeError about
    `home_score`, from the assign reaching for a column an empty frame has
    no room for, which reads like a schema bug rather than empty input.
    """
    with pytest.raises(ValueError, match="No games to score"):
        score_predictions(pd.DataFrame([]), _FixedMarginFitter(5.0))


def test_an_empty_frame_fails_the_same_way_the_brier_does() -> None:
    """Both entry points into scoring agree about what empty means."""
    from .brier import brier_score_df

    with pytest.raises(ValueError, match="No games to score"):
        brier_score_df(pd.DataFrame([]))
