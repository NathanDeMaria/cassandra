"""Tests for the scoring factory, and for the one method that has a scale."""

from datetime import datetime

import pytest
from endgame.types import Game

from .factory import get_scoring_function
from .sigmoid import DEFAULT_SIGMOID_SCALE


def _game(home_score: int, away_score: int) -> Game:
    return Game(
        home="Home",
        away="Away",
        home_score=home_score,
        away_score=away_score,
        neutral_site=False,
        completed=True,
        date=datetime(2026, 1, 1),
        game_id="1",
    )


def test_the_default_scale_is_what_it_always_was() -> None:
    """Every model published before the scale existed replayed at 10.

    A change here silently reinterprets every stored `scoring_method:
    sigmoid` -- the ratings would move and nothing in the artifact would say
    why.
    """
    assert DEFAULT_SIGMOID_SCALE == 10.0
    assert get_scoring_function("sigmoid")(_game(10, 0)) == pytest.approx(
        get_scoring_function("sigmoid", 10.0)(_game(10, 0))
    )


def test_a_smaller_scale_reads_a_short_margin_as_a_bigger_win() -> None:
    """The whole point: what a margin is worth depends on the sport's units.

    A three-set sweep at the points-scale default is 0.574 -- barely a win.
    At a scale the score line is actually measured in, it's a rout.
    """
    sweep = _game(3, 0)

    assert get_scoring_function("sigmoid", 10.0)(sweep) == pytest.approx(
        0.574, abs=1e-3
    )
    assert get_scoring_function("sigmoid", 1.0)(sweep) == pytest.approx(0.953, abs=1e-3)


def test_the_scale_stays_symmetric_about_a_tie() -> None:
    for scale in (0.5, 1.0, 10.0):
        score = get_scoring_function("sigmoid", scale)
        assert score(_game(0, 0)) == pytest.approx(0.5)
        assert score(_game(3, 0)) == pytest.approx(1 - score(_game(0, 3)))


@pytest.mark.parametrize("method", ["binary", "pythagorean"])
def test_the_methods_without_a_constant_ignore_the_scale(method: str) -> None:
    """`binary` reads only who won, and `pythagorean` divides the score line
    by itself -- so 3-0 in sets already reads the way 30-0 in points does."""
    at_default = get_scoring_function(method)(_game(3, 0))
    at_volleyball = get_scoring_function(method, 0.5)(_game(3, 0))

    assert at_default == at_volleyball


def test_an_unknown_method_still_says_so() -> None:
    with pytest.raises(ValueError, match="Unknown scoring method"):
        get_scoring_function("vibes")
