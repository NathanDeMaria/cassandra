"""Tests for the three matchup terms and how they compose."""

from collections.abc import Callable
from datetime import datetime

import pytest
from endgame.types import Game

from .adjustments import MatchupAdjustments, validated_travel_advantage
from .base_predictor import Predictor
from .compound import CompoundGlickoPredictor
from .elo import EloPredictor
from .epa import EpaIndex
from .glicko import GlickoPredictor
from .opponent_prior import OpponentPriorManager
from .qb_out import QbOutIndex
from .types import GameEpa

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


# ---------------------------------------------------------------- the update


def _elo(
    rest_advantage: float = 0.0,
    qb_out_penalty: float = 0.0,
    qb_out: QbOutIndex | None = None,
) -> EloPredictor:
    return EloPredictor(
        "test_league",
        rest_advantage=rest_advantage,
        qb_out_penalty=qb_out_penalty,
        qb_out=qb_out,
    )


def _glicko(
    rest_advantage: float = 0.0,
    qb_out_penalty: float = 0.0,
    qb_out: QbOutIndex | None = None,
) -> GlickoPredictor:
    return GlickoPredictor(
        "test_league",
        rest_advantage=rest_advantage,
        qb_out_penalty=qb_out_penalty,
        qb_out=qb_out,
        opponent_prior_manager=OpponentPriorManager("test_league"),
    )


def _rating(predictor: Predictor, team: str) -> float:
    """The team's rating as a number, whichever family the predictor is."""
    return predictor.ratings[team].rating


def _played(
    home: str, away: str, day: int, home_score: int, away_score: int, game_id: str
) -> Game:
    return Game(
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=False,
        completed=True,
        date=datetime(2026, 9, day),
        game_id=game_id,
    )


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_a_loss_without_the_quarterback_costs_the_rating_less(
    build: Callable[..., Predictor],
) -> None:
    """The update sees what the prediction saw.

    A side missing its starter was expected to do worse, so doing worse is
    less of a surprise and the rating moves less. Charging the loss in full
    would make the rating measure the team as it happened to be that week,
    and the returning starter would inherit it.
    """
    index = QbOutIndex({"g1": ["A"]})
    plain = build(qb_out_penalty=0.0, qb_out=index)
    priced = build(qb_out_penalty=80.0, qb_out=index)
    for p in (plain, priced):
        p.update_game(_played("A", "B", 1, 7, 28, game_id="g1"))

    assert _rating(priced, "A") > _rating(plain, "A")
    assert _rating(priced, "B") < _rating(plain, "B")


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_beating_a_rested_side_earns_more(build: Callable[..., Predictor]) -> None:
    """Same score, same opponent; the one who came off a bye was favoured."""
    plain = build(rest_advantage=0.0)
    priced = build(rest_advantage=30.0)
    for p in (plain, priced):
        p.update_game(_played("A", "B", 1, 21, 20, game_id="g1"))
        p.update_game(_played("A", "C", 8, 21, 20, game_id="g2"))
        # B has had two weeks; A played last week. A wins at B.
        p.update_game(_played("B", "A", 15, 20, 21, game_id="g3"))

    assert _rating(priced, "A") > _rating(plain, "A")


def test_the_compound_children_see_the_same_edge() -> None:
    """The unit contests are conditioned on the matchup like the parent is."""
    index = QbOutIndex({"g1": ["A"]})
    epa = EpaIndex({"g1": _game_epa(-0.2, 0.2)})
    plain = CompoundGlickoPredictor(
        "test_league", game_epa=epa, qb_out_penalty=0.0, qb_out=index
    )
    priced = CompoundGlickoPredictor(
        "test_league", game_epa=epa, qb_out_penalty=80.0, qb_out=index
    )
    for p in (plain, priced):
        p.update_game(_played("A", "B", 1, 7, 28, game_id="g1"))

    # A's offense was expected to struggle without him, so its poor day
    # counts against it less; B's defense had it easier, so its good day
    # counts for less.
    assert priced.get_units("A").offense.rating > plain.get_units("A").offense.rating
    assert priced.get_units("B").defense.rating < plain.get_units("B").defense.rating


def _game_epa(home: float, away: float) -> GameEpa:
    return GameEpa(
        home=home,
        away=away,
        home_plays=70,
        away_plays=70,
        home_weighted=home,
        away_weighted=away,
        home_weight=60.0,
        away_weight=60.0,
    )
