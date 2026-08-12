import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from endgame.types import Game

from .types import Matchup, Prediction, Rating


class RatingsUnsupported(NotImplementedError):
    """A predictor with no per-team ratings was asked for them.

    FlatPredictor is the case that matters: it predicts 0.5 for everyone and
    has no state to normalize. Raising is better than handing back an empty
    dict, which reads as "this model rates nobody" and is indistinguishable
    from a release that lost its ratings.
    """


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
    def state_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this predictor, as plain JSON types.

        Keys are the constructor's keyword arguments, so `from_state_dict` is
        usually just `cls(**data)`.
        """

    @classmethod
    @abstractmethod
    def from_state_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild a predictor from what `state_dict` emitted."""

    def save_state(self, path: Path) -> None:
        path.write_text(json.dumps(self.state_dict()))

    @classmethod
    def load_state(cls, path: Path) -> Self:
        """Read back a `save_state` file.

        The file is one serialization of `state_dict`, not a second format:
        callers that already hold the data -- a web service reading a
        ModelRelease, say -- should use `from_state_dict` and skip the round
        trip through a temp file.
        """
        return cls.from_state_dict(json.loads(path.read_text()))

    @property
    def ratings(self) -> dict[str, Rating]:
        """Per-team ratings, normalized. Raises for a predictor without any."""
        raise RatingsUnsupported(f"{type(self).__name__} has no team ratings")

    @classmethod
    def from_ratings(
        cls, league: str, ratings: Mapping[str, Rating], **params: Any
    ) -> Self:
        """Rebuild a predictor from normalized ratings and its params.

        The inverse of the `ratings` property, and the seam a consumer that
        holds a release -- ratings and params, no state file -- comes in
        through. `params` are the constructor's tuned keyword arguments
        (home_advantage, k, ...).
        """
        raise RatingsUnsupported(f"{cls.__name__} cannot be built from ratings")

    def pass_week(self) -> None:
        pass

    def pass_season(self) -> None:
        pass

    def postrun_callback(self) -> None:
        """Called after all seasons have been processed.

        This is the place for things like saving off final ratings."""
