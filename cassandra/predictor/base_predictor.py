from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from endgame.types import Game

from .types import Matchup, Prediction


class Predictor(ABC):
    def __init__(self, league: str) -> None:
        self._league = league

    @abstractmethod
    def predict_game(self, matchup: Matchup) -> Prediction:
        # A Matchup is only the pre-game half of a Game, so implementations
        # can't peek at the results.
        ...

    def update_game(self, game: Game) -> Prediction:
        """Predict and update internal state. Override in stateful subclasses."""
        return self.predict_game(game)

    @abstractmethod
    def save_state(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load_state(cls, path: Path) -> Self:
        pass

    def pass_week(self) -> None:
        pass

    def pass_season(self) -> None:
        pass

    def postrun_callback(self) -> None:
        """Called after all seasons have been processed.

        This is the place for things like saving off final ratings."""
