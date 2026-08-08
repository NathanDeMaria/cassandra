from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from bayes_opt import BayesianOptimization

type _ParameterBound = tuple[float, float] | Sequence[str]

# The share of the run that counts as "the end". If the best score was still
# climbing in here, the search hadn't flattened out by the time it stopped.
_TAIL_FRACTION = 0.25
# How much of the run's total gain has to land in the tail before that climb
# counts. Converged runs keep posting improvements in the last decimal place;
# without this they'd all read as "still going".
_MIN_TAIL_SHARE = 0.05
# How close to a bound the best point has to sit before that bound looks like
# the thing holding the search back, as a fraction of the bound's width.
_EDGE_FRACTION = 0.05
# The probes that count as "where the search ended up looking", so one lucky
# sample against a bound reads differently than a whole cluster of them.
_TOP_FRACTION = 0.1
_MIN_TOP = 3


@dataclass(frozen=True)
class _BoundHit:
    """The best point for one parameter landed against one of its bounds."""

    parameter: str
    edge: Literal["lower", "upper"]
    value: float
    bounds: tuple[float, float]
    # Share of the best-scoring probes that also sit in that same edge zone.
    top_share: float
    # A wider bound to try, ready to paste into the model config. None when the
    # hit edge can't be widened.
    suggestion: tuple[float, float] | None

    def __str__(self) -> str:
        low, high = self.bounds
        head = (
            f'"{self.parameter}": [{_number(low)}, {_number(high)}] put its best '
            f"point ({_number(self.value)}) against the {self.edge} bound, with "
            f"{self.top_share:.0%} of the best probes crowding the same edge"
        )
        if self.suggestion is None:
            # Only happens at a zero floor, where "wider" would mean negative.
            return (
                f"{head}: the search wants to go below {_number(low)}, so the "
                "parameter may want to be off entirely rather than rescaled"
            )
        new_low, new_high = self.suggestion
        return (
            f"{head}: try "
            f'"{self.parameter}": [{_number(new_low)}, {_number(new_high)}]'
        )


@dataclass(frozen=True)
class _Diagnostics:
    """Whether an optimization run stopped too early or searched too small a box."""

    evaluations: int
    # Index of the last evaluation that raised the best-so-far score.
    last_improvement: int
    # First index that counts as the tail of the run.
    tail_start: int
    # Share of the run's total gain that landed in the tail.
    tail_improvement_share: float
    # True when the best score was still meaningfully improving in the tail.
    still_improving: bool
    # An n_iter to try, when the caller said what the run used.
    suggested_iterations: int | None
    bound_hits: tuple[_BoundHit, ...]

    def warnings(self) -> list[str]:
        messages = []
        if self.still_improving:
            advice = (
                f'try "n_iter": {self.suggested_iterations}'
                if self.suggested_iterations is not None
                else "more iterations would probably have helped"
            )
            messages.append(
                f"the best score last improved at evaluation "
                f"{self.last_improvement + 1} of {self.evaluations}, with "
                f"{self.tail_improvement_share:.0%} of the total gain coming in the "
                f"final {self.evaluations - self.tail_start}: {advice}"
            )
        messages.extend(str(hit) for hit in self.bound_hits)
        return messages


def _diagnose(
    results: Sequence[Mapping[str, Any]],
    param_bounds: Mapping[str, _ParameterBound],
    *,
    iterations: int | None = None,
    tail_fraction: float = _TAIL_FRACTION,
    edge_fraction: float = _EDGE_FRACTION,
    min_tail_share: float = _MIN_TAIL_SHARE,
) -> _Diagnostics:
    """Judge, after the fact, whether a run deserved more iterations or wider bounds.

    `results` is the optimizer's probe log, in the order the probes were run:
    one mapping per evaluation with a "target" score and its "params".
    `iterations` is the n_iter the run was given, which turns "it needed longer"
    into a number to paste back into the model config.
    """
    if not results:
        raise ValueError("Cannot diagnose an optimization with no results")

    targets = [float(result["target"]) for result in results]
    ranked = sorted(range(len(targets)), key=targets.__getitem__, reverse=True)
    best_target = targets[ranked[0]]

    last_improvement = 0
    running_best = targets[0]
    for index, target in enumerate(targets[1:], start=1):
        if target > running_best:
            running_best = target
            last_improvement = index

    # At least one evaluation is always in the tail, so a one-probe run doesn't
    # divide by zero deciding whether it was still improving.
    tail_start = len(targets) - max(1, round(len(targets) * tail_fraction))
    total_gain = best_target - targets[0]
    tail_gain = best_target - max(targets[:tail_start], default=targets[0])
    tail_share = tail_gain / total_gain if total_gain > 0 else 0.0

    top = ranked[: max(_MIN_TOP, round(len(targets) * _TOP_FRACTION))]
    best_params = results[ranked[0]]["params"]

    return _Diagnostics(
        evaluations=len(targets),
        last_improvement=last_improvement,
        tail_start=tail_start,
        tail_improvement_share=tail_share,
        # A run whose very first probe was never beaten never had momentum to
        # extrapolate from, so more iterations aren't obviously worth it.
        still_improving=last_improvement >= tail_start and tail_share >= min_tail_share,
        # Doubling is the cheapest useful next try: the run that just stopped is
        # the only evidence available about how much longer it needed.
        suggested_iterations=None if iterations is None else iterations * 2,
        bound_hits=tuple(
            _find_bound_hits(
                best_params,
                [results[index]["params"] for index in top],
                param_bounds,
                edge_fraction,
            )
        ),
    )


def optimize(
    function: Callable[..., float],
    param_bounds: Mapping[str, _ParameterBound],
    iterations: int,
) -> tuple[float, dict[str, float | str]]:
    optimizer = BayesianOptimization(
        f=function,
        # The docs confirm list-of-str is how you do categorical
        # https://bayesian-optimization.github.io/BayesianOptimization/3.2.0/parameter_types.html#3.-Categorical-variables
        pbounds=param_bounds,  # ty: ignore[invalid-argument-type]
        random_state=1,
    )

    optimizer.maximize(n_iter=iterations)

    if optimizer.max is None:
        raise ValueError("Optimizer did not find a maximum")

    diagnostics = _diagnose(optimizer.res, param_bounds, iterations=iterations)
    for warning in diagnostics.warnings():
        print(f"[optimize] {warning}")

    return optimizer.max["target"], optimizer.max["params"]


def _find_bound_hits(
    best_params: Mapping[str, float | str],
    top_params: Sequence[Mapping[str, float | str]],
    param_bounds: Mapping[str, _ParameterBound],
    edge_fraction: float,
) -> Iterator[_BoundHit]:
    for name, (low, high) in _continuous_bounds(param_bounds):
        position = _position(best_params[name], low, high)
        if position is None:
            continue
        if position <= edge_fraction:
            edge: Literal["lower", "upper"] = "lower"
        elif position >= 1 - edge_fraction:
            edge = "upper"
        else:
            continue

        in_zone = sum(
            1
            for params in top_params
            if _in_edge_zone(_position(params[name], low, high), edge, edge_fraction)
        )
        yield _BoundHit(
            parameter=name,
            edge=edge,
            value=float(best_params[name]),
            bounds=(low, high),
            top_share=in_zone / len(top_params),
            suggestion=_widen((low, high), edge),
        )


def _widen(
    bounds: tuple[float, float], edge: Literal["lower", "upper"]
) -> tuple[float, float] | None:
    """Extend the crowded end outward by the bound's own width."""
    low, high = bounds
    width = high - low
    if edge == "upper":
        return (low, high + width)
    # A floor of exactly 0 is the author saying the parameter bottoms out there
    # (an increment, a count, a rate), so going negative isn't a suggestion
    # worth making.
    if low == 0:
        return None
    return (low - width, high)


def _continuous_bounds(
    param_bounds: Mapping[str, _ParameterBound],
) -> Iterator[tuple[str, tuple[float, float]]]:
    for name, bound in param_bounds.items():
        # Categoricals are the list of allowed strings: there's no "wider" for
        # them to go, so they can't be the reason a search came up short.
        if len(bound) != 2 or any(isinstance(value, str) for value in bound):
            continue
        low, high = bound
        yield name, (float(low), float(high))


def _number(value: float) -> str:
    """Render a bound the way it would be written in a model config."""
    return f"{value:g}"


def _position(value: float | str, low: float, high: float) -> float | None:
    """Where `value` falls in [low, high], as a fraction of the width."""
    if high <= low:
        return None
    return (float(value) - low) / (high - low)


def _in_edge_zone(
    position: float | None, edge: Literal["lower", "upper"], edge_fraction: float
) -> bool:
    if position is None:
        return False
    if edge == "lower":
        return position <= edge_fraction
    return position >= 1 - edge_fraction
