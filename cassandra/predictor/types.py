from typing import NamedTuple

from endgame.types import Game


class Prediction(NamedTuple):
    team1_win_prob: float


class GameResult(NamedTuple):
    prediction: Prediction
    game: Game
    year: int
    week_number: int
