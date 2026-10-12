import math

import pytest

from .distribution import (
    LogisticMargin,
    NormalMargin,
    collapse,
    decimal_logistic,
)


@pytest.mark.parametrize(
    "distribution",
    [LogisticMargin(3.0, 8.0, "points"), NormalMargin(3.0, 14.0, "points")],
    ids=["logistic", "normal"],
)
def test_the_win_prob_is_the_mass_above_zero(distribution) -> None:
    assert distribution.win_prob() == pytest.approx(1 - distribution.cdf(0))
    assert collapse(distribution).team1_win_prob == distribution.win_prob()


@pytest.mark.parametrize(
    "distribution",
    [LogisticMargin(-4.0, 8.0, "points"), NormalMargin(-4.0, 14.0, "points")],
    ids=["logistic", "normal"],
)
def test_half_the_mass_is_below_the_mean(distribution) -> None:
    assert distribution.cdf(distribution.mean) == pytest.approx(0.5)
    assert distribution.cdf(-100) < distribution.cdf(0) < distribution.cdf(100)


def test_a_logistic_sd_is_its_scale_times_pi_over_root_three() -> None:
    assert LogisticMargin(0.0, 3.0, "points").sd == pytest.approx(
        3.0 * math.pi / math.sqrt(3)
    )
    assert NormalMargin(0.0, 3.0, "points").sd == 3.0


def test_a_decimal_logistic_reads_a_gap_the_way_elo_does() -> None:
    """Ten to one at a gap of `scale`, which is what 400 has always meant."""
    assert decimal_logistic(400, 400, "rating").win_prob() == pytest.approx(10 / 11)
    assert decimal_logistic(-150, 400, "rating").win_prob() == pytest.approx(
        1 / (1 + 10 ** (150 / 400))
    )


def test_a_spread_reads_off_the_cdf() -> None:
    """Home favored by 7 at a mean of 7: covering is a coin flip."""
    margin = NormalMargin(7.0, 14.0, "points")
    assert 1 - margin.cdf(7.0) == pytest.approx(0.5)
    assert 1 - margin.cdf(3.0) > 0.5
