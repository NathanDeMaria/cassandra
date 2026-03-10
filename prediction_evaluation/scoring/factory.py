from typing import Callable

from endgame.types import Game

from .binary import binary_score
from .pythagorean import pythagorean_score
from .sigmoid import sigmoid_score

type ScoringFunction = Callable[[Game], float]


def get_scoring_function(name: str) -> ScoringFunction:
    """Factory function to get scoring functions by name."""
    if name == "binary":
        return binary_score
    if name == "pythagorean":
        return pythagorean_score
    if name == "sigmoid":
        return sigmoid_score
    raise ValueError(f"Unknown scoring method: {name}")
