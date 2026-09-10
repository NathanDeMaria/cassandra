from typing import Callable

from endgame.types import Game

from .binary import binary_score
from .pythagorean import pythagorean_score
from .sigmoid import DEFAULT_SIGMOID_SCALE, make_sigmoid_score

type ScoringFunction = Callable[[Game], float]


def get_scoring_function(
    name: str, sigmoid_scale: float = DEFAULT_SIGMOID_SCALE
) -> ScoringFunction:
    """A scoring function by name, with the sigmoid's scale if it takes one.

    `sigmoid_scale` is ignored by the two methods that have no such constant:
    `binary` reads only who won, and `pythagorean` is already scale-free --
    it divides the score line by itself, so 3-0 in sets reads exactly the way
    30-0 in points does. Only `sigmoid` has a divisor, and only there does
    this do anything.

    Accepted unconditionally rather than only for the sigmoid so callers
    don't have to know which methods take it: a predictor passes the
    parameter it holds and this decides whether it means anything, which is
    what lets `scoring_method` stay a searchable categorical.
    """
    if name == "binary":
        return binary_score
    if name == "pythagorean":
        return pythagorean_score
    if name == "sigmoid":
        return make_sigmoid_score(sigmoid_scale)
    raise ValueError(f"Unknown scoring method: {name}")
