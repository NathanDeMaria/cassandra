from datetime import datetime
from typing import Protocol

import pytest
from endgame.types import Game


class GameFactory(Protocol):
    """Builds the games a predictor test feeds in.

    A protocol rather than a Callable alias so the scores can keep their
    defaults: a test that only predicts a game doesn't care what it ended.
    """

    def __call__(
        self, home: str, away: str, home_score: int = 0, away_score: int = 0
    ) -> Game: ...


@pytest.fixture
def game() -> GameFactory:
    """A completed game between two teams, with the rest of Game filled in.

    Every predictor test needs one and none of them care about the fields
    that aren't the two names and the score, so the date and id are fixed
    here rather than in each file that wants a game.
    """

    def _game(home: str, away: str, home_score: int = 0, away_score: int = 0) -> Game:
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

    return _game
