from datetime import datetime
from pathlib import Path

import pytest
from endgame.types import Game

from .elo538 import Elo538Predictor


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


def test_elo538_save_load(tmp_path: Path) -> None:
    predictor = Elo538Predictor("test_league", home_advantage=90, k=25)
    predictor.predict_game(_game("Team A", "Team B", 3, 1))
    predictor.predict_game(_game("Team B", "Team C", 0, 2))

    save_path = tmp_path / "elo538.json"
    predictor.save_state(save_path)

    loaded = Elo538Predictor.load_state(save_path)
    assert loaded.get_rating("Team A") == pytest.approx(predictor.get_rating("Team A"))
    assert loaded.get_rating("Team B") == pytest.approx(predictor.get_rating("Team B"))
    assert loaded.get_rating("Team C") == pytest.approx(predictor.get_rating("Team C"))
    assert loaded.get_rating("Unknown") == 1500
