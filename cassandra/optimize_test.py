from typing import Any, Sequence

import pytest

from .optimize import _diagnose, optimize


def _results(
    targets: list[float], **params: Sequence[float | str]
) -> list[dict[str, Any]]:
    return [
        {
            "target": target,
            "params": {name: values[index] for name, values in params.items()},
        }
        for index, target in enumerate(targets)
    ]


@pytest.mark.parametrize("winner", ["a", "b", "c"])
def test_optimize__categorical(winner: str) -> None:
    param_name = "value"

    def _f(value: str, float_value: float) -> float:
        return float_value if winner == value else 0.0

    _, params = optimize(
        _f, {param_name: ("a", "b", "c"), "float_value": (0.0, 1.0)}, iterations=10
    )
    assert params[param_name] == winner


def test_diagnose__still_improving() -> None:
    # Every probe beats the last, so the search was cut off mid-climb.
    targets = [float(i) for i in range(20)]
    diagnostics = _diagnose(_results(targets, x=[0.5] * 20), {"x": (0.0, 1.0)})

    assert diagnostics.still_improving
    assert diagnostics.last_improvement == 19
    assert diagnostics.tail_improvement_share == pytest.approx(5 / 19)
    assert diagnostics.suggested_iterations is None
    assert any("more iterations" in w for w in diagnostics.warnings())


def test_diagnose__suggests_doubling_the_iterations() -> None:
    targets = [float(i) for i in range(20)]
    diagnostics = _diagnose(
        _results(targets, x=[0.5] * 20), {"x": (0.0, 1.0)}, iterations=60
    )

    assert diagnostics.suggested_iterations == 120
    assert any('try "n_iter": 120' in w for w in diagnostics.warnings())


def test_diagnose__converged_early() -> None:
    targets = [0.0, 1.0, 5.0] + [4.0] * 17
    diagnostics = _diagnose(_results(targets, x=[0.5] * 20), {"x": (0.0, 1.0)})

    assert not diagnostics.still_improving
    assert diagnostics.last_improvement == 2
    assert diagnostics.tail_improvement_share == 0.0
    assert not diagnostics.warnings()


def test_diagnose__trivial_tail_improvement() -> None:
    # Converged runs keep nudging the last decimal place; that isn't momentum.
    targets = [0.0] + [10.0] * 18 + [10.001]
    diagnostics = _diagnose(_results(targets, x=[0.5] * 20), {"x": (0.0, 1.0)})

    assert diagnostics.last_improvement == 19
    assert not diagnostics.still_improving
    assert diagnostics.tail_improvement_share < 0.01
    assert not diagnostics.warnings()


def test_diagnose__never_improved() -> None:
    # A flat run has no momentum to extrapolate from, so it isn't "still climbing".
    diagnostics = _diagnose(_results([1.0] * 10, x=[0.5] * 10), {"x": (0.0, 1.0)})

    assert not diagnostics.still_improving
    assert diagnostics.tail_improvement_share == 0.0


def test_diagnose__best_against_upper_bound() -> None:
    targets = [float(i) for i in range(10)]
    diagnostics = _diagnose(
        _results(targets, x=[1 + i / 9 for i in range(10)]), {"x": (1.0, 2.0)}
    )

    (hit,) = diagnostics.bound_hits
    assert hit.parameter == "x"
    assert hit.edge == "upper"
    assert hit.value == pytest.approx(2.0)
    # Of the top 3 probes (2, 17/9, 16/9), only the best is inside the edge zone.
    assert hit.top_share == pytest.approx(1 / 3)
    # The crowded end moves out by the bound's own width.
    assert hit.suggestion == (1.0, 3.0)
    assert any('try "x": [1, 3]' in w for w in diagnostics.warnings())


def test_diagnose__best_against_lower_bound() -> None:
    targets = [-float(i) for i in range(10)]
    diagnostics = _diagnose(
        _results(targets, x=[1 + i / 10 for i in range(10)]), {"x": (1.0, 2.0)}
    )

    (hit,) = diagnostics.bound_hits
    assert hit.edge == "lower"
    assert hit.value == pytest.approx(1.0)
    assert hit.suggestion == (0.0, 2.0)


def test_diagnose__no_suggestion_below_a_zero_floor() -> None:
    # Negative bounds are meaningless for the counts and rates these configs
    # hold, so a hit at a floor of 0 gets a different read.
    targets = [-float(i) for i in range(10)]
    diagnostics = _diagnose(
        _results(targets, x=[i / 10 for i in range(10)]), {"x": (0.0, 1.0)}
    )

    (hit,) = diagnostics.bound_hits
    assert hit.edge == "lower"
    assert hit.suggestion is None
    assert any("may want to be off entirely" in w for w in diagnostics.warnings())


def test_diagnose__interior_optimum_is_not_a_bound_hit() -> None:
    values = [i / 10 for i in range(10)]
    # Peaks in the middle of the range, which is what a well-sized bound looks like.
    targets = [-((value - 0.5) ** 2) for value in values]
    diagnostics = _diagnose(_results(targets, x=values), {"x": (0.0, 1.0)})

    assert not diagnostics.bound_hits


def test_diagnose__ignores_categoricals() -> None:
    # There's no wider bound to give a categorical, so it can never be flagged.
    diagnostics = _diagnose(
        _results([float(i) for i in range(6)], choice=["a", "b"] * 3),
        {"choice": ("a", "b")},
    )

    assert not diagnostics.bound_hits


def test_diagnose__requires_results() -> None:
    with pytest.raises(ValueError):
        _diagnose([], {"x": (0.0, 1.0)})


def test_diagnose__suggestion_stays_inside_what_the_model_accepts() -> None:
    """
    The case that cost a run: `season_regression` pinned at 0.75 with a
    [0, 0.75] range, and the suggestion doubled the width to [0, 1.5].
    `validated_regression` rejects anything above 1, so pasting that in
    trades a finished search for one that dies on its first probe.
    """
    targets = [float(i) for i in range(10)]
    diagnostics = _diagnose(
        _results(targets, season_regression=[0.75 * i / 9 for i in range(10)]),
        {"season_regression": (0.0, 0.75)},
    )

    (hit,) = diagnostics.bound_hits
    assert hit.edge == "upper"
    assert hit.suggestion == (0.0, 1.0)
    assert any('try "season_regression": [0, 1]' in w for w in diagnostics.warnings())


def test_diagnose__no_suggestion_at_the_top_of_the_domain() -> None:
    """
    Once the range already covers everything the parameter allows, the
    search asking for more isn't something a wider bound can answer.
    """
    targets = [float(i) for i in range(10)]
    diagnostics = _diagnose(
        _results(targets, season_regression=[i / 9 for i in range(10)]),
        {"season_regression": (0.0, 1.0)},
    )

    (hit,) = diagnostics.bound_hits
    assert hit.edge == "upper"
    assert hit.suggestion is None
    assert any("as high as this parameter goes" in w for w in diagnostics.warnings())


def test_diagnose__a_deviation_is_not_widened_below_zero() -> None:
    """
    `initial_rd` is a spread around a rating, so a negative floor is not a
    smaller starting uncertainty -- it's nothing at all.
    """
    targets = [-float(i) for i in range(10)]
    diagnostics = _diagnose(
        _results(targets, initial_rd=[80 + 27 * i for i in range(10)]),
        {"initial_rd": (80.0, 350.0)},
    )

    (hit,) = diagnostics.bound_hits
    assert hit.edge == "lower"
    # Clamped to the floor rather than the -190 a bare doubling would give.
    assert hit.suggestion == (0.0, 350.0)


def test_diagnose__an_unlisted_parameter_still_widens_freely() -> None:
    """
    `home_advantage` has no listed domain: a league where the road team is
    favored is odd, not impossible, so nothing should rule it out.
    """
    targets = [float(i) for i in range(10)]
    diagnostics = _diagnose(
        _results(targets, home_advantage=[100 + 10 * i for i in range(10)]),
        {"home_advantage": (100.0, 190.0)},
    )

    (hit,) = diagnostics.bound_hits
    assert hit.suggestion == (100.0, 280.0)
