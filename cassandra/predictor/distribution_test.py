import math

import numpy as np
import pytest

from .distribution import (
    DiscreteMargin,
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


def _three_or_seven() -> DiscreteMargin:
    """Home wins by 3 or 7, or loses by 3, in the shares below."""
    pmf = np.zeros(21)
    for margin, share in ((-3, 0.2), (3, 0.5), (7, 0.3)):
        pmf[margin + 10] = share
    return DiscreteMargin(-10, pmf)


def test_a_discrete_margin_reads_its_moments_and_cdf() -> None:
    margin = _three_or_seven()
    assert margin.mean == pytest.approx(-0.6 + 1.5 + 2.1)
    assert margin.cdf(3) == pytest.approx(0.7)
    assert margin.cdf(2.5) == pytest.approx(0.2)
    assert margin.cdf(-50) == 0.0
    assert margin.cdf(50) == pytest.approx(1.0)
    assert margin.win_prob() == pytest.approx(0.8)


def test_a_whole_number_line_can_push() -> None:
    """Home -3: a 3-point win is a push, a 7-point one covers."""
    assert _three_or_seven().p_cover(-3.0) == pytest.approx((0.3, 0.5, 0.2))


def test_a_half_point_line_cannot() -> None:
    assert _three_or_seven().p_cover(-3.5) == pytest.approx((0.3, 0.0, 0.7))
    assert _three_or_seven().p_cover(-2.5) == pytest.approx((0.8, 0.0, 0.2))


def test_a_home_underdog_covers_by_losing_close() -> None:
    """Home +3: losing by 3 pushes, and every win covers."""
    assert _three_or_seven().p_cover(3.0) == pytest.approx((0.8, 0.2, 0.0))
