from abc import ABC, abstractmethod

from endgame.types import Game

from .types import Prediction


class Predictor(ABC):
    def __init__(self, league: str) -> None:
        self._league = league

    @abstractmethod
    def predict_game(self, game: Game) -> Prediction:
        # I thought about just giving it team names
        # but then it makes things like a fake spread-based predictor
        # more difficult.
        # I'm trusting implementations to not peek at the results :)
        ...

    def pass_week(self) -> None:
        pass

    def pass_season(self) -> None:
        pass

    def postrun_callback(self) -> None:
        """Called after all seasons have been processed.

        This is the place for things like saving off final ratings."""
        pass
