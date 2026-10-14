from typing import Any, Self

from .base_predictor import Predictor
from .distribution import LogisticMargin, MarginDistribution
from .types import Matchup


class FlatPredictor(Predictor):
    """0.5 for everyone. The baseline every other model has to beat.

    It keeps no ratings, so `ratings` and `from_ratings` raise
    RatingsUnsupported -- inherited from Predictor, deliberately not
    overridden with something that pretends.
    """

    def predict_distribution(self, matchup: Matchup) -> MarginDistribution:
        return LogisticMargin(0.0, 1.0, "rating")

    def state_dict(self) -> dict[str, Any]:
        return {"league": self._league}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)
