import math
import signal
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from bayes_opt import BayesianOptimization

from .checkpoint import Checkpoint

type _ParameterBound = tuple[float, float] | Sequence[str]

# What each searchable parameter can be, whatever range a config asks for.
# Only `_widen` reads this, to keep the bound it suggests inside the values
# the predictors will accept.
#
# `season_regression` is the one with a real ceiling: `validated_regression`
# rejects anything outside [0, 1], because above 1 a team is reflected
# through its anchor rather than pulled toward it. The increments have a
# floor instead -- zero means "doesn't move", which is allowed. `initial_rd`
# looks like an increment and isn't: it's the deviation itself, which the
# Glicko update divides by, so zero is a ZeroDivisionError rather than a
# model that ignores its uncertainty. Its floor is the smallest value that
# still computes, not the smallest that makes sense -- this table is about
# what the predictors accept, and how wide a starting spread is *sensible*
# is the config's business.
#
# `home_advantage` is deliberately absent: a league where the road team is
# favored is strange, not impossible, so nothing here should rule it out.
_DOMAINS: Mapping[str, tuple[float, float]] = {
    "season_regression": (0.0, 1.0),
    "k": (0.0, math.inf),
    "initial_rd": (1.0, math.inf),
    "weekly_rd_increase": (0.0, math.inf),
    "season_rd_increase": (0.0, math.inf),
}

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
        if self.suggestion is None and self.edge == "lower":
            # A floor there's nothing below, so "wider" would mean negative.
            return (
                f"{head}: the search wants to go below {_number(low)}, so the "
                "parameter may want to be off entirely rather than rescaled"
            )
        if self.suggestion is None:
            # Already at the most this parameter accepts. Widening isn't the
            # answer, and offering a bound the predictors reject would cost
            # the next run rather than improve it.
            return (
                f"{head}: {_number(high)} is as high as this parameter goes, so "
                "the search is asking for something the model can't do"
            )
        new_low, new_high = self.suggestion
        return (
            f'{head}: try "{self.parameter}": [{_number(new_low)}, {_number(new_high)}]'
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


# bayes_opt's own default: random probes before the first fitted suggestion.
# Named because the checkpoint arithmetic counts them -- a search of
# `n_iter` is `INIT_POINTS + n_iter` probes -- and a resumed attempt must
# not draw them again.
INIT_POINTS = 5

# Probes between saves. A probe on ncaafb is five to ten seconds of replay,
# so this is a couple of minutes of work at risk between saves -- the SIGTERM
# save below is what covers the gap when the reclaim is announced -- against
# a state dump and an upload that cost about a second.
CHECKPOINT_EVERY = 25


def optimize(
    function: Callable[..., float],
    param_bounds: Mapping[str, _ParameterBound],
    iterations: int,
    checkpoint: Checkpoint | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> tuple[float, dict[str, float | str]]:
    """Search `param_bounds` for `iterations` probes past the random start.

    With a `checkpoint`, the search saves itself after every
    `checkpoint_every` probes and on SIGTERM, and starts by loading whatever
    that checkpoint holds -- so a container that is reclaimed and retried
    picks up where the last attempt got to rather than from zero. See
    `cassandra.checkpoint`. The result is the same either way: chunked
    probing with a reload between chunks reproduces a single `maximize`
    exactly, which `optimize_test` asserts.
    """
    optimizer = BayesianOptimization(
        f=function,
        # The docs confirm list-of-str is how you do categorical
        # https://bayesian-optimization.github.io/BayesianOptimization/3.2.0/parameter_types.html#3.-Categorical-variables
        pbounds=param_bounds,
        random_state=1,
    )
    total = INIT_POINTS + iterations

    resumed = _load(optimizer, checkpoint)
    if resumed:
        print(f"[optimize] resumed at probe {resumed} of {total}")
    with _saving_on_sigterm(optimizer, checkpoint):
        while len(optimizer.res) < total:
            done = len(optimizer.res)
            # The random start only once, and never on a resumed search:
            # a reload carries those probes with it.
            init_points = INIT_POINTS if done == 0 else 0
            step = min(checkpoint_every, total - done)
            optimizer.maximize(init_points=init_points, n_iter=step - init_points)
            _save(optimizer, checkpoint)

    if optimizer.max is None:
        raise ValueError("Optimizer did not find a maximum")

    diagnostics = _diagnose(optimizer.res, param_bounds, iterations=iterations)
    for warning in diagnostics.warnings():
        print(f"[optimize] {warning}")

    # Only once the result is in hand: a search that raised above keeps its
    # save for the retry, and one that finished has nothing left to resume.
    if checkpoint is not None:
        checkpoint.clear()
    return optimizer.max["target"], optimizer.max["params"]


def _load(optimizer: BayesianOptimization, checkpoint: Checkpoint | None) -> int:
    """Restore a save into `optimizer`; the number of probes it carried, 0 if none."""
    if checkpoint is None:
        return 0
    state = checkpoint.load()
    if state is None:
        return 0
    # `load_state` wants a path, not bytes.
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        handle.write(state)
        handle.flush()
        optimizer.load_state(handle.name)
    return len(optimizer.res)


def _save(optimizer: BayesianOptimization, checkpoint: Checkpoint | None) -> None:
    if checkpoint is None:
        return
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        optimizer.save_state(handle.name)
        checkpoint.save(Path(handle.name).read_bytes())


@contextmanager
def _saving_on_sigterm(
    optimizer: BayesianOptimization, checkpoint: Checkpoint | None
) -> Iterator[None]:
    """Save the search when the container is told to stop.

    ECS delivers SIGTERM ahead of the kill when a spot instance is reclaimed,
    which is the moment the save is worth the most: the probes since the
    last periodic save are still only in memory. The handler saves and then
    exits with the conventional 128 + 15, and Batch's retry -- which keys on
    the host failure, not the exit code -- brings the next attempt up on the
    save.

    Only the main thread can set a signal handler, and only when there is a
    checkpoint to save to is there anything to do; otherwise the default
    disposition stays.
    """
    if checkpoint is None or threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, frame: object) -> None:
        print("[optimize] SIGTERM: saving the search before stopping", flush=True)
        _save(optimizer, checkpoint)
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


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
            suggestion=_widen(name, (low, high), edge),
        )


def _widen(
    parameter: str, bounds: tuple[float, float], edge: Literal["lower", "upper"]
) -> tuple[float, float] | None:
    """Extend the crowded end outward by the bound's own width.

    Clamped to what the parameter can actually be. A suggestion outside that
    is worse than none at all: it reads as advice, and following it costs a
    run -- `"season_regression": [0, 1.5]` is rejected by
    `validated_regression` on the first probe, so the search dies at startup
    instead of finishing wider.
    """
    low, high = bounds
    width = high - low
    domain_low, domain_high = _DOMAINS.get(parameter, (-math.inf, math.inf))
    if edge == "upper":
        # Already as high as the parameter goes, so there's nothing to widen
        # into -- the search wanting more means something else is wrong.
        if high >= domain_high:
            return None
        return (low, min(high + width, domain_high))
    # A floor of exactly 0 is the author saying the parameter bottoms out there
    # (an increment, a count, a rate), so going negative isn't a suggestion
    # worth making.
    if low == 0 or low <= domain_low:
        return None
    return (max(low - width, domain_low), high)


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
