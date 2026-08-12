from datetime import datetime

import pytest
from endgame.types import Game

from .elo import EloPredictor
from .matchup import predict_matchup


def _predictor() -> EloPredictor:
    return EloPredictor(
        "test_league",
        home_advantage=100,
        ratings={"Team A": 1600, "Team B": 1500},
    )


def test_matchup_matches_what_the_same_game_would_get() -> None:
    """The stand-in has to be indistinguishable from a real Game."""
    predictor = _predictor()
    game = Game(
        home="Team A",
        away="Team B",
        home_score=0,
        away_score=0,
        neutral_site=False,
        completed=False,
        date=datetime(2026, 1, 1),
        game_id="1",
    )

    assert predict_matchup(
        predictor, "Team A", "Team B", date=datetime(2026, 1, 1)
    ).team1_win_prob == pytest.approx(predictor.predict_game(game).team1_win_prob)


def test_home_advantage_applies_unless_the_site_is_neutral() -> None:
    predictor = _predictor()

    home = predict_matchup(predictor, "Team A", "Team B").team1_win_prob
    neutral = predict_matchup(
        predictor, "Team A", "Team B", neutral_site=True
    ).team1_win_prob

    assert home > neutral > 0.5


def test_the_probability_is_the_home_teams() -> None:
    """Swapping the two sides has to swap the number, not repeat it."""
    predictor = _predictor()

    assert predict_matchup(
        predictor, "Team B", "Team A", neutral_site=True
    ).team1_win_prob == pytest.approx(
        1
        - predict_matchup(
            predictor, "Team A", "Team B", neutral_site=True
        ).team1_win_prob
    )
