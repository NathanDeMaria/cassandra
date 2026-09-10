import math
from typing import Callable

from endgame.types import Game

# Points of margin per unit of logit: the divisor that decides how much of a
# blowout a scoring function is allowed to notice.
#
# 10 because that is what this was hardcoded at for every league in every
# sport, so a model that doesn't name a scale replays exactly as it always
# did. It is a default, not a finding -- `BlendedGlickoPredictor` says as
# much about the same number: "10 was never fitted to anything".
#
# It is also the number that makes this scoring method only as good as its
# assumption about the score line. 10 is a point margin's scale, where a
# blowout is 20 or 30 and the logistic has room to separate one from a
# one-possession game. It does not travel:
#
#     league    margin        sigmoid(margin / 10) reaches
#     ncaafb    +/- 20-30     0.88 .. 0.95
#     nhl       +/- 1-5       0.52 .. 0.62
#     ncaawvb   +/- 1-3 sets  0.52 .. 0.57
#
# Replayed over 16,630 ncaawvb matches, glicko's brier by scale, against
# 0.250 for predicting 0.5 every time and 0.153 for `pythagorean`:
#
#     scale   10      2       1       0.5
#     brier   0.230   0.171   0.155   0.154
#
# At 10 a 3-0 sweep and a 3-2 nailbiter are almost the same evidence, so the
# method is barely beating the coin flip. At a scale the score line is
# actually measured in it is the equal of anything else on offer. Nothing
# about the method was wrong; the constant was.
#
# Which is why the scale is a parameter now rather than this constant. Every
# league's `glicko_full.json` searches it over a range in its own units.
DEFAULT_SIGMOID_SCALE = 10.0


def make_sigmoid_score(scale: float = DEFAULT_SIGMOID_SCALE) -> Callable[[Game], float]:
    """A scorer that reads a margin as the share of a game it stands for.

    `scale` is in the units of whatever the league's score line is -- points,
    goals, sets -- so it is the one number that has to change when the sport
    does. `BlendedGlickoPredictor.mov_scale` is the same quantity under a
    different name, applied to the same logistic; the two default to the same
    number on purpose.

    Built per call rather than kept as a module-level instance, because the
    scale is now something a search moves.
    """

    def _score(game: Game) -> float:
        point_diff = game.home_score - game.away_score
        return 1 / (1 + math.exp(-point_diff / scale))

    return _score
