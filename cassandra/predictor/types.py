from datetime import datetime
from typing import NamedTuple, Protocol

from endgame.types import Game


class Matchup(Protocol):
    """The part of a game that's known *before* it's played.

    `Game` satisfies this structurally, so callers can pass one straight
    through, but a `predict_game` implementation that takes a `Matchup`
    can't reach the scores.
    """

    @property
    def home(self) -> str: ...

    @property
    def away(self) -> str: ...

    @property
    def neutral_site(self) -> bool: ...

    @property
    def date(self) -> datetime: ...

    @property
    def game_id(self) -> str: ...


class Prediction(NamedTuple):
    team1_win_prob: float


class GameResult(NamedTuple):
    prediction: Prediction
    game: Game
    year: int
    week_number: int
