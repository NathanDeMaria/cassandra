import math
from datetime import datetime
from typing import Any, NamedTuple

import pytest

from .conftest import GameFactory
from .margin_elo import _LOGIT_PER_ELO, MarginEloPredictor


class _matchup(NamedTuple):
    """A game that hasn't been played, which is all `expected_margin` needs.

    `Game` is the only thing conftest builds and it's always at a venue, so
    the neutral-site case needs a Matchup of its own.
    """

    home: str
    away: str
    neutral_site: bool = False
    date: datetime = datetime(2023, 1, 1)
    game_id: str = "1"


# What every predictor does -- ratings, anchors, regression, save/load -- is
# in contract_test.py. What's left here is what makes this one different:
# it learns from the margin rather than from the result.


def _predictor(**kwargs: Any) -> MarginEloPredictor:
    defaults: dict[str, Any] = {
        "home_advantage": 0,
        "k": 0.1,
        "margin_scale": 10.0,
        "mov_cap": 40.0,
    }
    return MarginEloPredictor("test_league", **{**defaults, **kwargs})


def test_the_rating_gap_reads_as_a_margin(game: GameFactory) -> None:
    """The exchange rate, stated once: an Elo gap is a number of points.

    A 400-point gap is a 10-to-1 favorite by construction, which at a
    margin_scale of 10 is ln(10) * 10 points.
    """
    predictor = _predictor(anchors={"Good": 1700, "Bad": 1300})

    margin = predictor.expected_margin(game("Good", "Bad"))

    assert margin == pytest.approx(400 * _LOGIT_PER_ELO * 10)
    assert margin == pytest.approx(math.log(10) * 10)


def test_the_win_probability_is_the_expected_margin_restated(
    game: GameFactory,
) -> None:
    """What keeps the rest of the pipeline able to consume this model.

    Everything downstream reads `team1_win_prob`, so the margin has to reach
    it as a probability -- and get back out again through the prob->margin
    fit, which it does because the two are the same logistic.
    """
    predictor = _predictor(anchors={"Good": 1600, "Bad": 1450})
    matchup = game("Good", "Bad")

    prediction = predictor.predict_game(matchup)

    margin = predictor.expected_margin(matchup)
    assert prediction.team1_win_prob == pytest.approx(1 / (1 + math.exp(-margin / 10)))


def test_beating_the_number_moves_a_team_up(game: GameFactory) -> None:
    """The whole point: a team that wins by more than expected gains."""
    predictor = _predictor(anchors={"Good": 1700, "Bad": 1300})
    expected = predictor.expected_margin(game("Good", "Bad"))

    predictor.update_game(game("Good", "Bad", 100, 100 - round(expected) - 10))

    assert predictor.get_rating("Good") > 1700
    assert predictor.get_rating("Bad") < 1300


def test_winning_by_less_than_expected_costs_the_favorite(
    game: GameFactory,
) -> None:
    """Where this model and every other one here disagree.

    A 30-point favorite that wins by 2 has told us something bad about
    itself. `binary_score` reads that game as a win and moves the favorite
    *up*; this reads it as a 28-point miss and moves it down.
    """
    predictor = _predictor(anchors={"Good": 1700, "Bad": 1300})

    predictor.update_game(game("Good", "Bad", 72, 70))

    assert predictor.get_rating("Good") < 1700
    assert predictor.get_rating("Bad") > 1300


def test_a_result_exactly_on_the_number_moves_nobody(game: GameFactory) -> None:
    # A 400-point gap is one unit of logit, so this scale makes the expected
    # margin exactly 10 -- a scoreline a game can actually finish at.
    predictor = _predictor(
        anchors={"Good": 1700, "Bad": 1300}, margin_scale=10 / math.log(10)
    )
    assert predictor.expected_margin(game("Good", "Bad")) == pytest.approx(10.0)

    predictor.update_game(game("Good", "Bad", 90, 80))

    assert predictor.get_rating("Good") == pytest.approx(1700)
    assert predictor.get_rating("Bad") == pytest.approx(1300)


def test_the_move_is_zero_sum(game: GameFactory) -> None:
    """Rating is relative, so what one side gains the other has to lose.

    Without this the league's average rating drifts, and every anchor -- and
    every prior fit against one -- slowly stops meaning what it did.
    """
    predictor = _predictor()

    predictor.update_game(game("Team A", "Team B", 90, 70))

    assert predictor.get_rating("Team A") - 1500 == pytest.approx(
        1500 - predictor.get_rating("Team B")
    )


def test_k_scales_the_move(game: GameFactory) -> None:
    slow = _predictor(k=0.05)
    fast = _predictor(k=0.10)

    slow.update_game(game("Team A", "Team B", 90, 70))
    fast.update_game(game("Team A", "Team B", 90, 70))

    assert fast.get_rating("Team A") - 1500 == pytest.approx(
        2 * (slow.get_rating("Team A") - 1500)
    )


def test_a_point_of_surprise_is_worth_k_points_of_expected_margin(
    game: GameFactory,
) -> None:
    """What `k` means, checked in the units it's meant to be read in.

    The rating scale is an implementation detail of this model; the number a
    config author is choosing is how much of a 10-point surprise the model
    keeps.
    """
    predictor = _predictor(k=0.1)
    predictor.update_game(game("Team A", "Team B", 80, 70))

    # Both teams moved, so the gap moved by twice the one-sided delta.
    assert predictor.expected_margin(game("Team A", "Team B")) == pytest.approx(2.0)


def test_a_blowout_past_the_cap_counts_as_the_cap(game: GameFactory) -> None:
    """Garbage time isn't evidence.

    The margin a basketball game finishes at depends on how long the
    starters stayed in, so past some point the extra points say more about
    the winning coach than the winning team.
    """
    capped = _predictor(mov_cap=30.0)
    at_the_cap = _predictor(mov_cap=30.0)

    capped.update_game(game("Team A", "Team B", 130, 70))
    at_the_cap.update_game(game("Team A", "Team B", 100, 70))

    assert capped.get_rating("Team A") == pytest.approx(at_the_cap.get_rating("Team A"))


def test_the_cap_works_on_a_loss_too(game: GameFactory) -> None:
    predictor = _predictor(mov_cap=30.0)
    at_the_cap = _predictor(mov_cap=30.0)

    predictor.update_game(game("Team A", "Team B", 70, 130))
    at_the_cap.update_game(game("Team A", "Team B", 70, 100))

    assert predictor.get_rating("Team A") == pytest.approx(
        at_the_cap.get_rating("Team A")
    )


def test_home_advantage_is_a_number_of_points(game: GameFactory) -> None:
    """Quoted in points rather than rating, which is also what keeps it
    independent of `margin_scale` -- see the invariance test below.
    """
    predictor = MarginEloPredictor("test_league", home_advantage=3.5, margin_scale=10.0)

    assert predictor.expected_margin(game("Team A", "Team B")) == pytest.approx(3.5)
    assert predictor.expected_margin(
        _matchup("Team A", "Team B", neutral_site=True)
    ) == pytest.approx(0.0)


def test_the_home_team_has_to_beat_the_home_number_to_gain(
    game: GameFactory,
) -> None:
    """A home win by less than the home advantage is a bad result."""
    predictor = MarginEloPredictor(
        "test_league", home_advantage=6.0, k=0.1, margin_scale=10.0
    )

    predictor.update_game(game("Team A", "Team B", 71, 70))

    assert predictor.get_rating("Team A") < 1500


@pytest.mark.parametrize("bad", [0.0, -3.0])
def test_a_margin_scale_that_cannot_divide_is_rejected(bad: float) -> None:
    """Every prediction divides by it, so the config is where this should fail.

    A search handed a range reaching 0 otherwise reports a ZeroDivisionError
    from inside a probe, hours from the file that asked for it.
    """
    with pytest.raises(ValueError, match="margin_scale"):
        MarginEloPredictor("test_league", margin_scale=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_cap_that_flattens_every_game_is_rejected(bad: float) -> None:
    """At a cap of 0 every game is a tie and the model is the flat baseline."""
    with pytest.raises(ValueError, match="mov_cap"):
        MarginEloPredictor("test_league", mov_cap=bad)


def test_the_predicted_margins_do_not_depend_on_margin_scale(
    game: GameFactory,
) -> None:
    """The property that makes `margin_scale` a probability parameter only.

    Both models are handed the same schedule and disagree about what a
    margin is worth as a win probability. They still expect the identical
    margin in the next game -- the rating scale stretches and the points
    come out the same. That is what lets a `margin_mae` config pin
    `margin_scale` and spend its probes on the parameters the objective can
    actually see.

    Teams that start level, which is the case the invariance is exact in;
    `test_anchors_are_the_one_thing_margin_scale_still_reaches` covers the
    other one.
    """
    narrow = _predictor(margin_scale=8.0)
    wide = _predictor(margin_scale=16.0)
    schedule = [
        ("Team A", "Team B", 80, 70),
        ("Team C", "Team A", 60, 75),
        ("Team B", "Team C", 90, 88),
    ]
    for home, away, home_score, away_score in schedule:
        narrow.update_game(game(home, away, home_score, away_score))
        wide.update_game(game(home, away, home_score, away_score))

    upcoming = game("Team A", "Team C")
    assert narrow.expected_margin(upcoming) == pytest.approx(
        wide.expected_margin(upcoming)
    )
    # ... and the win probabilities they read off that margin do differ,
    # which is the half a brier run is there to fit.
    assert narrow.predict_game(upcoming).team1_win_prob != pytest.approx(
        wide.predict_game(upcoming).team1_win_prob
    )


def test_anchors_are_the_one_thing_margin_scale_still_reaches(
    game: GameFactory,
) -> None:
    """Where the invariance above stops, and why a league can have an opinion.

    An anchor is quoted on the rating scale, so the gap between two of them
    buys more points at a wider `margin_scale`. It's the only channel left --
    a league whose teams all start at the mean has none of it -- and it is
    what keeps a pinned `margin_scale` a decision worth checking rather than
    an arbitrary constant.
    """
    anchors = {"Big": 1600, "Small": 1300}
    narrow = _predictor(margin_scale=8.0, anchors=anchors)
    wide = _predictor(margin_scale=16.0, anchors=anchors)

    first_meeting = game("Big", "Small")

    assert wide.expected_margin(first_meeting) == pytest.approx(
        2 * narrow.expected_margin(first_meeting)
    )
