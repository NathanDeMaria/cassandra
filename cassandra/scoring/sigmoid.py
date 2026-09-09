import math
from typing import Callable

from endgame.types import Game


def _make_sigmoid_score(k: float = 10) -> Callable[[Game], float]:
    def _score(game: Game) -> float:
        point_diff = game.home_score - game.away_score
        return 1 / (1 + math.exp(-point_diff / k))

    return _score


# k is in the units of whatever the league's score line is, which makes this
# scoring method only as good as that assumption. 10 is a point margin's
# scale -- basketball, football, where a blowout is 20 or 30 and the sigmoid
# has room to separate one from a one-possession game.
#
# It does not travel to a sport scored in something else. A volleyball match
# is decided by at most three sets, so every result ncaawvb can produce maps
# into [0.43, 0.57] and the model learns almost nothing from the score line:
# replayed untuned over 16,630 matches, glicko scores 0.153 on `pythagorean`
# and 0.230 on `sigmoid`, against 0.250 for predicting 0.5 every time.
# `pythagorean` has no such constant in it and reads a 3-0 as 1.0, which is
# why ncaawvb's config offers that and not this.
#
# TODO: fiddle with k -- and if it ever becomes a tuned parameter rather than
# a constant, that is what would make this usable across score lines.
sigmoid_score = _make_sigmoid_score(k=10)
