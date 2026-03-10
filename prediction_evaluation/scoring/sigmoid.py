import math
from typing import Callable

from endgame.types import Game


def _make_sigmoid_score(k: float = 10) -> Callable[[Game], float]:
    def _score(game: Game) -> float:
        point_diff = game.home_score - game.away_score
        return 1 / (1 + math.exp(-point_diff / k))

    return _score


# TODO: fiddle with k
sigmoid_score = _make_sigmoid_score(k=10)
