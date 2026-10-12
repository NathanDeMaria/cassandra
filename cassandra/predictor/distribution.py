"""What a predictor says about a game's margin, before it is a win probability.

Every model here already reasons about a distribution of the home side's
margin and then throws all of it away but one number. The Elo and Glicko
families take a logistic of a rating gap: that is the survival function at
zero of a logistic margin centred on the gap. The margin Glicko's implied
prediction is a Gaussian in points, with a spread from the observation noise
and both ratings' deviations. `Predictor.predict_distribution` hands the
distribution out whole, and `Predictor.predict_game` is `collapse` of it, so
the win probability the pipeline reads is unchanged.

Units
-----

A distribution says which axis it is on. `"points"` is the scoreboard: a
model that measures its gap in points (the margin Elo, the margin Glicko and
its children) can be read against a spread directly, `cdf(-spread)` being the
chance the home side fails to cover. `"rating"` is a model's own rating
scale, where only the sign of the margin means anything to a bettor: those
models still need the prob->margin fit to say how many points a gap is.

What it isn't yet
-----------------

Continuous, so there are no ties and no key numbers: `win_prob` is P(margin
> 0) and a push has no mass. And it is the distribution the model's *win
probability* implies, not one fitted to margins: the logistic scales were
searched against brier. Both are the next step's problem, and this module is
the seam it comes in through.
"""

import math
from typing import Literal, NamedTuple, Protocol

from .types import Prediction

MarginUnits = Literal["points", "rating"]

_LN10 = math.log(10)
_SQRT2 = math.sqrt(2)
_LOGISTIC_SD = math.pi / math.sqrt(3)


class MarginDistribution(Protocol):
    """The home side's margin of victory, as a distribution."""

    @property
    def units(self) -> MarginUnits: ...

    @property
    def mean(self) -> float: ...

    @property
    def sd(self) -> float: ...

    def cdf(self, margin: float) -> float:
        """P(home margin <= `margin`)."""
        ...

    def win_prob(self) -> float:
        """P(home margin > 0): what `collapse` reads."""
        ...


class LogisticMargin(NamedTuple):
    """A logistic margin: the shape every Elo and Glicko win probability has.

    `scale` is the natural one, so a rating model whose probability is
    `1 / (1 + 10 ** (-gap / s))` has `scale = s / ln 10`.
    """

    loc: float
    scale: float
    units: MarginUnits

    @property
    def mean(self) -> float:
        return self.loc

    @property
    def sd(self) -> float:
        return self.scale * _LOGISTIC_SD

    def cdf(self, margin: float) -> float:
        return 1 / (1 + math.exp((self.loc - margin) / self.scale))

    def win_prob(self) -> float:
        return 1 / (1 + math.exp(-self.loc / self.scale))


class NormalMargin(NamedTuple):
    """A Gaussian margin: the margin Glicko's implied prediction."""

    loc: float
    scale: float
    units: MarginUnits

    @property
    def mean(self) -> float:
        return self.loc

    @property
    def sd(self) -> float:
        return self.scale

    def cdf(self, margin: float) -> float:
        return _normal_cdf((margin - self.loc) / self.scale)

    def win_prob(self) -> float:
        return _normal_cdf(self.loc / self.scale)


def decimal_logistic(gap: float, scale: float, units: MarginUnits) -> LogisticMargin:
    """The logistic whose win probability is `1 / (1 + 10 ** (-gap / scale))`.

    How the Elo and Glicko families have always quoted their logistic: odds
    of ten to one at a gap of `scale`.
    """
    return LogisticMargin(gap, scale / _LN10, units)


def collapse(distribution: MarginDistribution) -> Prediction:
    """The distribution as the win probability the pipeline reads."""
    return Prediction(team1_win_prob=distribution.win_prob())


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / _SQRT2))
