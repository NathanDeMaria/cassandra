"""What a search's box is, and whether a point sits in it.

Split out of `optimize` because the check runs in two places: there, on
the seeds handed to a search, and in `OptimizationConfig`, on the seeds a
config declares -- and a config is read by the serving side, which must
not drag bayes_opt and sklearn in behind it (`release_test` checks).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: The marker a config puts third in a bound to search whole numbers:
#: `"passes": [1, 4, "int"]`. bayes_opt takes the type `int` there, which
#: JSON can't carry, so the config spells it and `for_bayes_opt` translates.
INTEGER: Literal["int"] = "int"
#: The marker for a bound searched on a log scale: `"player_rating_weight":
#: [1, 40, "log"]`, for a knob whose effect is a ratio -- 2 to 4 matters about
#: as much as 20 to 40 -- so that the bottom of a wide range gets searched as
#: closely as the top. See `Scale`.
LOG: Literal["log"] = "log"

type ParameterBound = (
    tuple[float, float]
    | tuple[int, int, Literal["int"]]
    | tuple[float, float, Literal["log"]]
    | Sequence[str]
)
#: A point in a search's box: one value per searched parameter.
type Seed = Mapping[str, float | str]


def continuous_range(bound: ParameterBound) -> tuple[float, float] | None:
    """The `(low, high)` of a numeric bound, whole-numbered or not; None for a categorical."""
    if len(bound) == 3 and bound[2] in (INTEGER, LOG):
        return float(bound[0]), float(bound[1])  # type: ignore[arg-type]
    if len(bound) == 2 and not any(isinstance(value, str) for value in bound):
        return float(bound[0]), float(bound[1])  # type: ignore[arg-type]
    return None


def is_integer(bound: ParameterBound) -> bool:
    return len(bound) == 3 and bound[2] == INTEGER


def is_log(bound: ParameterBound) -> bool:
    return len(bound) == 3 and bound[2] == LOG


# Where a log scale starts, as a share of the bound's width above its low
# end: `[0, 20, "log"]` is searched as log(x + 0.2). Zero has no logarithm,
# and zero is often the value that means "off", so the scale is shifted to
# start a little above it rather than refusing it -- the first 1% of the
# width gets as much of the search as the decade above it.
LOG_OFFSET = 0.01


@dataclass(frozen=True)
class Scale:
    """How one numeric bound maps onto the [0, 1] that bayes_opt searches.

    bayes_opt fits one length scale to every dimension in the units it is
    handed, so a box that mixes `initial_rd: [200, 800]` with
    `glicko_share: [0.55, 2]` gets a length scale of a few hundred, and
    across that the shares and regressions are all the same point: the
    surrogate cannot see them move, and the search spends its probes in the
    corners of the wide dimensions. Searching each bound as a fraction of
    its width puts every one on the same footing.
    """

    low: float
    high: float
    log: bool = False

    def _warp(self, value: float) -> float:
        if not self.log:
            return value
        return math.log(value - self.low + LOG_OFFSET * (self.high - self.low))

    def to_unit(self, value: float) -> float:
        low, high = self._warp(self.low), self._warp(self.high)
        return (self._warp(value) - low) / (high - low)

    def from_unit(self, unit: float) -> float:
        """Rounded to twelve significant figures, so a seed comes back as
        itself -- 188.5 rather than 188.49999999999997 -- and a published fit
        reads as the seed it held rather than a neighbour of it."""
        low, high = self._warp(self.low), self._warp(self.high)
        warped = low + unit * (high - low)
        if self.log:
            value = math.exp(warped) + self.low - LOG_OFFSET * (self.high - self.low)
        else:
            value = warped
        return min(self.high, max(self.low, float(f"{value:.12g}")))

    def saved(self) -> list[float | str]:
        """How a checkpoint records it; `Scale.from_saved` reads it back."""
        return [self.low, self.high, LOG] if self.log else [self.low, self.high]

    @staticmethod
    def from_saved(saved: Sequence[float | str]) -> "Scale":
        return Scale(float(saved[0]), float(saved[1]), len(saved) == 3)


def scale(bound: ParameterBound) -> Scale | None:
    """How `bound` is searched as [0, 1]; None for a bound searched as itself.

    Whole-numbered bounds keep their own units -- bayes_opt rounds them in
    its kernel, which only works on the integers themselves, and they span a
    few steps, close to a unit already -- and so do categoricals, which it
    one-hot encodes. A bound with no width has nothing to scale.
    """
    span = continuous_range(bound)
    if span is None or is_integer(bound) or span[1] <= span[0]:
        return None
    return Scale(span[0], span[1], is_log(bound))


def for_bayes_opt(param_bounds: Mapping[str, ParameterBound]) -> dict[str, Any]:
    """The box as `BayesianOptimization(pbounds=)` takes it, see `scale`."""
    return {
        name: (
            (0.0, 1.0)
            if scale(bound) is not None
            else (bound[0], bound[1], int)
            if is_integer(bound)
            else bound
        )
        for name, bound in param_bounds.items()
    }


def to_unit(point: Seed, param_bounds: Mapping[str, ParameterBound]) -> dict[str, Any]:
    """`point` in the units `for_bayes_opt` searches."""
    out: dict[str, Any] = {}
    for name, value in point.items():
        found = scale(param_bounds[name])
        out[name] = value if found is None else found.to_unit(float(value))
    return out


def from_unit(
    point: Mapping[str, Any], param_bounds: Mapping[str, ParameterBound]
) -> dict[str, Any]:
    """The inverse of `to_unit`: a point bayes_opt proposed, in the config's units."""
    out: dict[str, Any] = {}
    for name, value in point.items():
        found = scale(param_bounds[name])
        out[name] = value if found is None else found.from_unit(float(value))
    return out


def misplaced(seed: Seed, param_bounds: Mapping[str, ParameterBound]) -> str | None:
    """Why `seed` can't be probed in `param_bounds`, or None when it can.

    A seed names every searched parameter and nothing else, a continuous
    one sits inside its bounds and a categorical one is one of the choices.
    bayes_opt only warns about a point outside the box, and a seed that
    is silently outside it is exactly the kind of "the search started from
    the previous fit" that didn't.
    """
    missing = sorted(set(param_bounds) - set(seed))
    if missing:
        return f"missing {', '.join(missing)}"
    extra = sorted(set(seed) - set(param_bounds))
    if extra:
        return f"not searched: {', '.join(extra)}"
    for name, bound in param_bounds.items():
        value = seed[name]
        span = continuous_range(bound)
        if span is None:
            if value not in bound:
                return f"{name}={value!r} is not one of {', '.join(map(str, bound))}"
            continue
        low, high = span
        if isinstance(value, str) or not low <= float(value) <= high:
            return f"{name}={value!r} is outside [{low:g}, {high:g}]"
        if is_integer(bound) and not float(value).is_integer():
            return f"{name}={value!r} is not a whole number"
    return None
