"""Tests for the margin-observation Glicko.

The rating half -- anchors, regression, round trips, a win moves the winner
up -- is checked against every model in contract_test.py. What is here is
the step itself, which is a Kalman filter and can be checked against the
arithmetic, and the two things the parent does differently: the prediction
and the smoother.
"""

import math
from datetime import datetime
from typing import Any, NamedTuple

import pytest

from .conftest import GameFactory
from .glicko import _Rating
from .margin_glicko import MarginGlickoPredictor


class _matchup(NamedTuple):
    """A game that hasn't been played; conftest's are always at a venue."""

    home: str
    away: str
    neutral_site: bool = False
    date: datetime = datetime(2023, 1, 1)
    game_id: str = "1"


def _predictor(**kwargs: Any) -> MarginGlickoPredictor:
    # Home advantage 0 and a round exchange rate, so the expected margins in
    # the tests below can be read off the ratings by hand.
    kwargs.setdefault("home_advantage", 0)
    kwargs.setdefault("points_per_rating", 0.1)
    kwargs.setdefault("initial_rd", 100)
    kwargs.setdefault("obs_sd", 10)
    return MarginGlickoPredictor("test_league", **kwargs)


def _kalman(
    rd: float, opp_rd: float, obs_sd: float, c: float, innovation: float
) -> tuple[float, float]:
    """The step as the module docstring writes it, for a Gaussian."""
    observation = obs_sd**2 / c**2 + opp_rd**2
    gain = rd**2 / (rd**2 + observation)
    return gain * innovation / c, math.sqrt(rd**2 * (1 - gain))


def test_the_step_is_the_kalman_update(game: GameFactory) -> None:
    """One game, checked against the algebra rather than against itself.

    Two level teams at rd 100 and 0.1 points per rating unit: the home side
    wins by 20, a 20-point surprise. Each side's observation noise is the
    margin's 10 points -- 100 rating units squared -- plus the opponent's
    own deviation squared, so the gain is 100^2 / (100^2 + 100^2 + 100^2).
    """
    predictor = _predictor()
    predictor.update_game(game("A", "B", 20, 0))

    move, rd = _kalman(100, 100, 10, 0.1, 20)
    assert move == pytest.approx(200 / 3)
    assert predictor.get_rating("A").rating == pytest.approx(1500 + move)
    assert predictor.get_rating("B").rating == pytest.approx(1500 - move)
    assert predictor.get_rating("A").rating_deviation == pytest.approx(rd)
    assert predictor.get_rating("B").rating_deviation == pytest.approx(rd)


def test_a_result_that_matched_the_expectation_moves_nothing(
    game: GameFactory,
) -> None:
    """The surprise is measured from the ratings' expected margin, edge included."""
    predictor = _predictor(
        home_advantage=30, ratings={"A": _Rating(1600, 50), "B": _Rating(1500, 50)}
    )
    # A is 100 up plus 30 of home edge, at 0.1 points a unit: expected +13.
    predictor.update_game(game("A", "B", 13, 0))

    assert predictor.get_rating("A").rating == pytest.approx(1600)
    assert predictor.get_rating("B").rating == pytest.approx(1500)
    # The deviation still shrinks: a confirming result is still a measurement.
    assert predictor.get_rating("A").rating_deviation < 50


def test_a_surprise_is_linear_in_the_gaussian(game: GameFactory) -> None:
    """Twice the surprise, twice the move: no squash and no saturation.

    This is the property the sigmoid target lacked -- a 40-point blowout was
    worth little more than a 20-point win -- and the reason the model exists.
    """
    small, big = _predictor(), _predictor()
    small.update_game(game("A", "B", 20, 0))
    big.update_game(game("A", "B", 40, 0))

    assert big.get_rating("A").rating - 1500 == pytest.approx(
        2 * (small.get_rating("A").rating - 1500)
    )


def test_an_unmeasured_opponent_is_weaker_evidence(game: GameFactory) -> None:
    """Glicko's g(RD), by another road: the opponent's rd is observation noise."""
    sure = _predictor(ratings={"B": _Rating(1500, 10)})
    unsure = _predictor(ratings={"B": _Rating(1500, 300)})
    sure.update_game(game("A", "B", 20, 0))
    unsure.update_game(game("A", "B", 20, 0))

    assert unsure.get_rating("A").rating < sure.get_rating("A").rating
    # And A stays less sure of itself after the less informative game.
    assert (
        unsure.get_rating("A").rating_deviation > sure.get_rating("A").rating_deviation
    )


def test_a_t_discounts_a_big_surprise_and_leaves_a_small_one(
    game: GameFactory,
) -> None:
    """The robust step trusts a surprise less the further out it sits.

    At nu = 4 a result inside the noise moves the rating almost as the
    Gaussian would; one five standard deviations out moves it a fraction as
    far. Under the Gaussian both are on the same line.
    """
    for margin in (5, 100):
        gaussian, robust = _predictor(), _predictor(nu=4)
        gaussian.update_game(game("A", "B", margin, 0))
        robust.update_game(game("A", "B", margin, 0))
        ratio = (robust.get_rating("A").rating - 1500) / (
            gaussian.get_rating("A").rating - 1500
        )
        if margin == 5:
            assert ratio > 0.9
        else:
            assert ratio < 0.5


def test_a_large_nu_is_the_gaussian(game: GameFactory) -> None:
    gaussian, nearly = _predictor(), _predictor(nu=1e9)
    gaussian.update_game(game("A", "B", 45, 3))
    nearly.update_game(game("A", "B", 45, 3))

    assert nearly.get_rating("A") == pytest.approx(gaussian.get_rating("A"))


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_scale_of_zero_or_less_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="obs_sd"):
        MarginGlickoPredictor("test_league", obs_sd=value)
    with pytest.raises(ValueError, match="nu"):
        MarginGlickoPredictor("test_league", nu=value)
    with pytest.raises(ValueError, match="points_per_rating"):
        MarginGlickoPredictor("test_league", points_per_rating=value)
    with pytest.raises(ValueError, match="prediction_sd"):
        MarginGlickoPredictor("test_league", prediction_sd=value)


def test_the_prediction_is_the_gaussian_win_probability(game: GameFactory) -> None:
    """Phi of the expected margin over the innovation sd, both teams' rd in."""
    predictor = _predictor(
        home_advantage=20, ratings={"A": _Rating(1600, 30), "B": _Rating(1500, 40)}
    )
    # Expected margin (100 + 20) * 0.1 = 12; sd^2 = 10^2 + 0.1^2 (30^2 + 40^2).
    sd = math.sqrt(100 + 0.01 * (900 + 1600))
    expected = 0.5 * (1 + math.erf(12 / sd / math.sqrt(2)))

    assert predictor.predict_game(game("A", "B")).team1_win_prob == pytest.approx(
        expected
    )


def test_an_unsure_rating_predicts_closer_to_even(game: GameFactory) -> None:
    sure = _predictor(ratings={"A": _Rating(1600, 10), "B": _Rating(1500, 10)})
    unsure = _predictor(ratings={"A": _Rating(1600, 200), "B": _Rating(1500, 200)})

    assert (
        0.5
        < unsure.predict_game(game("A", "B")).team1_win_prob
        < sure.predict_game(game("A", "B")).team1_win_prob
    )


def test_a_fixed_prediction_sd_ignores_the_deviations(game: GameFactory) -> None:
    fixed = _predictor(
        prediction_sd=16, ratings={"A": _Rating(1600, 200), "B": _Rating(1500, 200)}
    )
    expected = 0.5 * (1 + math.erf(10 / 16 / math.sqrt(2)))

    assert fixed.predict_game(game("A", "B")).team1_win_prob == pytest.approx(expected)


def test_neutral_sites_get_no_edge(game: GameFactory) -> None:
    predictor = _predictor(home_advantage=50)
    sited = predictor.predict_game(game("A", "B")).team1_win_prob
    neutral = predictor.predict_game(
        _matchup("A", "B", neutral_site=True)
    ).team1_win_prob

    assert sited > 0.5
    assert neutral == pytest.approx(0.5)


def _week(predictor: MarginGlickoPredictor, game: GameFactory, *games: tuple) -> None:
    for index, (home, away, home_score, away_score) in enumerate(games):
        predictor.update_game(
            game(home, away, home_score, away_score, game_id=str(index))
        )
    predictor.pass_week()


def test_the_smoother_re_rates_a_win_against_what_the_opponent_became(
    game: GameFactory,
) -> None:
    """The parent's `passes`, replayed with the margin step.

    A beats B by one in week 1. In week 2 D, a nobody, beats B by 49. The
    filter leaves A's credit where it was; the smoother re-prices A's win
    against the B that week 2 revealed and takes it back -- and then some,
    because on the margin a one-point win over a team that bad is a bad
    result, which is the difference from the parent's binary version of
    this test. A still stands above B.
    """
    filter_ = _predictor(passes=1)
    smoothed = _predictor(passes=3)
    for predictor in (filter_, smoothed):
        _week(predictor, game, ("A", "B", 21, 20))
        _week(predictor, game, ("D", "B", 49, 0))

    assert smoothed.get_rating("A").rating < filter_.get_rating("A").rating
    assert smoothed.get_rating("A").rating > smoothed.get_rating("B").rating


def test_the_smoother_moves_only_the_means(game: GameFactory) -> None:
    filter_ = _predictor(passes=1)
    smoothed = _predictor(passes=4)
    for predictor in (filter_, smoothed):
        _week(predictor, game, ("A", "B", 21, 20), ("C", "D", 3, 0))
        _week(predictor, game, ("D", "B", 49, 0), ("A", "C", 7, 6))

    for team in "ABCD":
        assert smoothed.get_rating(team).rating_deviation == pytest.approx(
            filter_.get_rating(team).rating_deviation
        )


def test_the_state_carries_this_models_knobs_and_not_the_parents(
    game: GameFactory,
) -> None:
    predictor = _predictor(nu=6, prediction_sd=15, passes=2)
    predictor.update_game(game("A", "B", 20, 3))

    state = predictor.state_dict()
    for parents in ("k", "scoring_method", "sigmoid_scale", "prediction_scale"):
        assert parents not in state
    assert state["obs_sd"] == 10
    assert state["nu"] == 6
    assert state["prediction_sd"] == 15
    assert state["points_per_rating"] == 0.1

    restored = MarginGlickoPredictor.from_state_dict(state)
    assert restored.nu == 6
    assert restored.predict_game(game("A", "B")).team1_win_prob == pytest.approx(
        predictor.predict_game(game("A", "B")).team1_win_prob
    )


def test_a_gaussian_round_trips_its_absent_nu(game: GameFactory) -> None:
    predictor = _predictor()
    state = predictor.state_dict()
    assert state["nu"] is None
    assert MarginGlickoPredictor.from_state_dict(state).nu is None
