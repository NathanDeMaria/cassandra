import pytest

from .optimize import optimize


@pytest.mark.parametrize("winner", ["a", "b", "c"])
def test_optimize__categorical(winner: str) -> None:
    param_name = "value"

    def _f(value: str, float_value: float) -> float:
        return float_value if winner == value else 0.0

    _, params = optimize(
        _f, {param_name: ("a", "b", "c"), "float_value": (0.0, 1.0)}, iterations=10
    )
    assert params[param_name] == winner
