"""What a search's box is, and whether a point sits in it.

Split out of `optimize` because the check runs in two places: there, on
the seeds handed to a search, and in `OptimizationConfig`, on the seeds a
config declares -- and a config is read by the serving side, which must
not drag bayes_opt and sklearn in behind it (`release_test` checks).
"""

from collections.abc import Mapping, Sequence

type ParameterBound = tuple[float, float] | Sequence[str]
#: A point in a search's box: one value per searched parameter.
type Seed = Mapping[str, float | str]


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
        if len(bound) != 2 or any(isinstance(choice, str) for choice in bound):
            if value not in bound:
                return f"{name}={value!r} is not one of {', '.join(map(str, bound))}"
            continue
        low, high = bound
        if isinstance(value, str) or not float(low) <= float(value) <= float(high):
            return f"{name}={value!r} is outside [{float(low):g}, {float(high):g}]"
    return None
