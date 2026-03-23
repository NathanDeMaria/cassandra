from datetime import datetime
from pathlib import Path

from endgame.types import Game

from .flat import FlatPredictor


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


def test_flat_save_load(tmp_path: Path) -> None:
    predictor = FlatPredictor("test_league")
    save_path = tmp_path / "flat.json"
    predictor.save_state(save_path)

    loaded = FlatPredictor.load_state(save_path)
    assert loaded.predict_game(_game("A", "B", 1, 0)).team1_win_prob == 0.5
