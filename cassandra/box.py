"""What a search's box is, and whether a point sits in it.

Split out of `optimize` because the check runs in two places: there, on
the seeds handed to a search, and in `OptimizationConfig`, on the seeds a
config declares -- and a config is read by the serving side, which must
not drag bayes_opt and sklearn in behind it (`release_test` checks).
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

#: The marker a config puts third in a bound to search whole numbers:
#: `"passes": [1, 4, "int"]`. bayes_opt takes the type `int` there, which
#: JSON can't carry, so the config spells it and `for_bayes_opt` translates.
INTEGER: Literal["int"] = "int"

type ParameterBound = (
    tuple[float, float] | tuple[int, int, Literal["int"]] | Sequence[str]
)
#: A point in a search's box: one value per searched parameter.
type Seed = Mapping[str, float | str]


def continuous_range(bound: ParameterBound) -> tuple[float, float] | None:
    """The `(low, high)` of a numeric bound, whole-numbered or not; None for a categorical."""
    if len(bound) == 3 and bound[2] == INTEGER:
        return float(bound[0]), float(bound[1])  # type: ignore[arg-type]
    if len(bound) == 2 and not any(isinstance(value, str) for value in bound):
        return float(bound[0]), float(bound[1])  # type: ignore[arg-type]
    return None


def is_integer(bound: ParameterBound) -> bool:
    return len(bound) == 3 and bound[2] == INTEGER


def for_bayes_opt(param_bounds: Mapping[str, ParameterBound]) -> dict[str, Any]:
    """The box as `BayesianOptimization(pbounds=)` takes it."""
    return {
        name: (bound[0], bound[1], int) if is_integer(bound) else bound
        for name, bound in param_bounds.items()
    }


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
