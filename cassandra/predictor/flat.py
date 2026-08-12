from typing import Any, Self

from .base_predictor import Predictor
from .types import Matchup, Prediction


class FlatPredictor(Predictor):
    """0.5 for everyone. The baseline every other model has to beat.

    It keeps no ratings, so `ratings` and `from_ratings` raise
    RatingsUnsupported -- inherited from Predictor, deliberately not
    overridden with something that pretends.
    """

    def predict_game(self, matchup: Matchup) -> Prediction:
        return Prediction(team1_win_prob=0.5)

    def state_dict(self) -> dict[str, Any]:
        return {"league": self._league}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)
