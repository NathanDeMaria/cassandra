"""Tests for the three matchup terms and how they compose."""

from datetime import datetime

import pytest
from endgame.types import Game

from .adjustments import MatchupAdjustments, validated_travel_advantage
from .qb_out import QbOutIndex

_HOME = "Ohio State Buckeyes"
_AWAY = "USC Trojans"  # a long way from Columbus
_NEAR = "Michigan Wolverines"  # not far at all
_UNKNOWN = "Nobody State Nobodies"


class _Matchup:
    def __init__(
        self, home: str, away: str, neutral_site: bool = False, game_id: str = "g1"
    ) -> None:
        self.home, self.away = home, away
        self.neutral_site = neutral_site
        self.game_id = game_id
        self.date = datetime(2026, 9, 12)


def _game(home: str, away: str, day: int) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=1,
        away_score=0,
        neutral_site=False,
        completed=True,
        date=datetime(2026, 9, day),
        game_id="g0",
    )


def test_everything_off_is_worth_nothing() -> None:
    """The default is the model that shipped before any of this."""
    assert MatchupAdjustments().points(_Matchup(_HOME, _AWAY)) == 0.0


def test_a_longer_trip_is_worth_more_to_the_home_team() -> None:
    adj = MatchupAdjustments(travel_advantage=5.0)

    far = adj.travel_points(_Matchup(_HOME, _AWAY))
    near = adj.travel_points(_Matchup(_HOME, _NEAR))

    assert far > near > 0


def test_travel_is_per_thousand_kilometres() -> None:
    """So the weight is a number a person can hold."""
    from cassandra.travel import distance_km

    adj = MatchupAdjustments(travel_advantage=5.0)
    km = distance_km(_AWAY, _HOME)

    assert km is not None
    assert adj.travel_points(_Matchup(_HOME, _AWAY)) == pytest.approx(5.0 * km / 1000)


def test_no_travel_term_at_a_neutral_site() -> None:
    """A `Game` says a game was neutral and never says where it was."""
    adj = MatchupAdjustments(travel_advantage=5.0)

    assert adj.travel_points(_Matchup(_HOME, _AWAY, neutral_site=True)) == 0.0


def test_an_unknown_venue_is_worth_nothing_rather_than_a_guess() -> None:
    adj = MatchupAdjustments(travel_advantage=5.0)

    assert adj.travel_points(_Matchup(_HOME, _UNKNOWN)) == 0.0
    assert adj.travel_points(_Matchup(_UNKNOWN, _AWAY)) == 0.0


def test_a_missing_quarterback_costs_the_side_that_is_missing_him() -> None:
    index = QbOutIndex({"g1": [_AWAY]})
    adj = MatchupAdjustments(qb_out_penalty=40.0, qb_out=index)

    assert adj.qb_points(_Matchup(_HOME, _AWAY)) == pytest.approx(40.0)
    # And the sign flips when it's the home side.
    flipped = MatchupAdjustments(
        qb_out_penalty=40.0, qb_out=QbOutIndex({"g1": [_HOME]})
    )
    assert flipped.qb_points(_Matchup(_HOME, _AWAY)) == pytest.approx(-40.0)


def test_two_missing_quarterbacks_are_nobody_s_edge() -> None:
    index = QbOutIndex({"g1": [_HOME, _AWAY]})
    adj = MatchupAdjustments(qb_out_penalty=40.0, qb_out=index)

    assert adj.qb_points(_Matchup(_HOME, _AWAY)) == 0.0


def test_a_game_the_index_never_saw_assumes_both_are_fine() -> None:
    """Which is every fixture, and the whole live-prediction default."""
    adj = MatchupAdjustments(qb_out_penalty=40.0, qb_out=QbOutIndex({"g1": [_AWAY]}))

    assert adj.qb_points(_Matchup(_HOME, _AWAY, game_id="not-played-yet")) == 0.0


def test_the_qb_term_still_applies_at_a_neutral_site() -> None:
    """Unlike travel: a bowl doesn't give anyone their quarterback back."""
    adj = MatchupAdjustments(qb_out_penalty=40.0, qb_out=QbOutIndex({"g1": [_AWAY]}))

    assert adj.qb_points(_Matchup(_HOME, _AWAY, neutral_site=True)) == pytest.approx(
        40.0
    )


def test_the_three_terms_add() -> None:
    """All signed toward the home team, so a sign error shows up alone."""
    adj = MatchupAdjustments(
        rest_advantage=5.0,
        travel_advantage=5.0,
        qb_out_penalty=40.0,
        qb_out=QbOutIndex({"g1": [_AWAY]}),
    )
    adj.record(_game(_HOME, _AWAY, 1))
    adj.record(_game(_AWAY, _NEAR, 8))  # the away side plays again; home rests

    matchup = _Matchup(_HOME, _AWAY)
    total = adj.points(matchup)

    assert total == pytest.approx(
        adj.rest.adjustment(matchup)
        + adj.travel_points(matchup)
        + adj.qb_points(matchup)
    )
    assert adj.rest.adjustment(matchup) > 0
    assert total > adj.qb_points(matchup)


def test_the_season_boundary_clears_only_what_is_stateful() -> None:
    """Travel is a lookup and the QB index is built offline; only rest forgets."""
    adj = MatchupAdjustments(rest_advantage=5.0, travel_advantage=5.0)
    adj.record(_game(_HOME, _AWAY, 1))
    adj.record(_game(_AWAY, _NEAR, 8))
    matchup = _Matchup(_HOME, _AWAY)
    assert adj.rest.adjustment(matchup) != 0

    adj.pass_season()

    assert adj.rest.adjustment(matchup) == 0.0
    assert adj.travel_points(matchup) > 0


@pytest.mark.parametrize("value", [-0.1, -10.0])
def test_a_negative_travel_weight_is_refused(value: float) -> None:
    """It would be the claim that a long flight helps."""
    with pytest.raises(ValueError, match="travel_advantage must be non-negative"):
        validated_travel_advantage(value)
