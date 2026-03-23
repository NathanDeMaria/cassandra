from datetime import datetime
from pathlib import Path

import pytest
from endgame.types import Game

from .glicko import GlickoPredictor


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


def test_glicko_save_load(tmp_path: Path) -> None:
    predictor = GlickoPredictor(
        "test_league",
        home_advantage=100,
        k=60,
        weekly_rd_increase=2,
        season_rd_increase=100,
        initial_rd=200,
        scoring_method="binary",
    )
    predictor.update_game(_game("Team A", "Team B", 1, 0))
    predictor.update_game(_game("Team C", "Team A", 2, 1))
    predictor.pass_week()

    save_path = tmp_path / "glicko.json"
    predictor.save_state(save_path)

    loaded = GlickoPredictor.load_state(save_path)

    for team in ("Team A", "Team B", "Team C"):
        orig = predictor.get_rating(team)
        restored = loaded.get_rating(team)
        assert restored.rating == pytest.approx(orig.rating)
        assert restored.rating_deviation == pytest.approx(orig.rating_deviation)

    unknown = loaded.get_rating("Unknown")
    assert unknown.rating == 1500
    assert unknown.rating_deviation == 200
