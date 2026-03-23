from datetime import datetime
from pathlib import Path

import pytest
from endgame.types import Game

from .elo import EloPredictor


def _game(home: str, away: str, home_score: int, away_score: int) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=False,
        completed=True,
        date=datetime(2023, 1, 1),
        game_id="1",
    )


def test_elo():
    predictor = EloPredictor("placeholder")
    team_a = "Team A"
    team_b = "Team B"
    original_rating_a = predictor.get_rating(team_a)
    original_rating_b = predictor.get_rating(team_b)
    predictor.predict_game(_game(team_a, team_b, 1, 0))
    assert (
        predictor.get_rating(team_a)
        > original_rating_a
        == original_rating_b
        > predictor.get_rating(team_b)
    )


def test_elo_save_load(tmp_path: Path) -> None:
    predictor = EloPredictor("test_league", home_advantage=80, k=15)
    predictor.predict_game(_game("Team A", "Team B", 2, 1))
    predictor.predict_game(_game("Team C", "Team A", 0, 3))

    save_path = tmp_path / "elo.json"
    predictor.save_state(save_path)

    loaded = EloPredictor.load_state(save_path)
    assert loaded.get_rating("Team A") == pytest.approx(predictor.get_rating("Team A"))
    assert loaded.get_rating("Team B") == pytest.approx(predictor.get_rating("Team B"))
    assert loaded.get_rating("Team C") == pytest.approx(predictor.get_rating("Team C"))
    assert loaded.get_rating("Unknown") == 1500


def test_save_load_predictions_match(tmp_path: Path) -> None:
    """After loading, predictions for new games match the original predictor."""
    predictor = EloPredictor("test_league")
    predictor.predict_game(_game("Team A", "Team B", 1, 0))

    save_path = tmp_path / "elo.json"
    predictor.save_state(save_path)
    loaded = EloPredictor.load_state(save_path)

    new_game = _game("Team A", "Team B", 2, 1)
    assert loaded.predict_game(new_game).team1_win_prob == pytest.approx(
        predictor.predict_game(new_game).team1_win_prob
    )
