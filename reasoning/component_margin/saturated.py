"""The published units model, its record stepped against a margin that saturates.

The record measures a margin against `mu = c * gap`, linear however lopsided
the game. Scoreboards aren't: past a few touchdowns the starters sit and the
clock runs, and a 45-point favorite wins by 6.6 less than linear on average
(2002-2021, `mismatch_efficiency.py`), so every such game marks the favorite
down -- by enough to show in its next competitive game (+0.6 points,
`mismatch_next.py`). The size of each step is right (its surprise doesn't
predict the next one); its expectation is what's off.

So the step's expectation is `f(mu) = M tanh(mu / M)` and the step is the
extended Kalman filter's: the observation's slope at the expectation,
`c f'(mu) = c (1 - tanh^2(mu / M))`, takes the place of `c`. A lopsided game
is measured against what lopsided games end at, and says less about the gap
because the margin moves less with it there. `mismatch_scale` is M in
points; None is the published model exactly.

Only the record. A per-play reading doesn't saturate the same way -- the
read gap tracked the expected one at every size of gap, `diag_scale.py` --
so the units step as before. The prediction is the published logistic,
unchanged: a saturating expectation moves what is learned, and the win
probability of a game the scale already calls 99% hardly moves.
"""

import math
from typing import Any

from cassandra.predictor.glicko import _Rating
from cassandra.predictor.unit_glicko import UnitMarginGlickoPredictor


class SaturatedUnitMarginPredictor(UnitMarginGlickoPredictor):
    def __init__(self, league: str, *, mismatch_scale: float | None = None, **kwargs: Any) -> None:
        super().__init__(league, **kwargs)
        self._mismatch_scale = mismatch_scale or None

    def _step(self, my: _Rating, opp: _Rating, margin: float, home_adjustment: float) -> _Rating:
        if self._mismatch_scale is None:
            return super()._step(my, opp, margin, home_adjustment)
        c = self._points_per_rating
        m_scale = self._mismatch_scale
        linear = c * (my.rating + home_adjustment - opp.rating)
        squashed = math.tanh(linear / m_scale)
        expected = m_scale * squashed
        slope = c * max(1.0 - squashed * squashed, 1e-6)
        noise = self._obs_sd**2
        observation = noise / slope**2 + opp.rating_deviation**2
        gain = my.rating_deviation**2 / (my.rating_deviation**2 + observation)
        return _Rating(
            my.rating + gain * (margin - expected) / slope,
            math.sqrt(my.rating_deviation**2 * (1 - gain)),
        )
