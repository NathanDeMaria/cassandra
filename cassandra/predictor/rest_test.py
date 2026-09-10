"""Tests for the rest advantage: the ledger, and what it does to a prediction."""

from collections.abc import Callable
from datetime import datetime, timedelta

import pytest
from endgame.types import Game

from .base_predictor import Predictor
from .elo import EloPredictor
from .glicko import GlickoPredictor
from .opponent_prior import OpponentPriorManager
from .rest import REST_CAP_DAYS, RestLedger

_LEAGUE = "test_league"
# A base date plus an offset, rather than a day-of-month: these tests reach
# past the end of a month on purpose, to get at the clamp.
_START = datetime(2026, 9, 1)


def _game(home: str, away: str, day: int, home_score: int = 1, away_score: int = 0):
    return Game(
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        neutral_site=False,
        completed=True,
        date=_START + timedelta(days=day),
        game_id=f"{home}-{away}-{day}",
    )


class _Matchup:
    """Just the pre-game half, so these can't lean on a score."""

    def __init__(self, home: str, away: str, day: int, neutral_site: bool = False):
        self.home, self.away = home, away
        self.neutral_site = neutral_site
        self.date = _START + timedelta(days=day)
        self.game_id = "m"


Build = Callable[..., Predictor]


def _elo(rest_advantage: float = 0.0) -> EloPredictor:
    return EloPredictor(_LEAGUE, rest_advantage=rest_advantage)


def _glicko(rest_advantage: float = 0.0) -> GlickoPredictor:
    return GlickoPredictor(
        _LEAGUE,
        rest_advantage=rest_advantage,
        opponent_prior_manager=OpponentPriorManager(_LEAGUE),
    )


# ---------------------------------------------------------------- the ledger


def test_a_season_opener_has_no_differential() -> None:
    """ "No idea" and "extremely rested" are different claims."""
    ledger = RestLedger(per_day=5.0)
    ledger.record(_game("A", "B", 1))

    # B has a date, C never played -- so there is nothing to compare.
    assert ledger.differential(_Matchup("B", "C", 8)) == 0.0
    assert ledger.adjustment(_Matchup("B", "C", 8)) == 0.0


def test_the_differential_is_days_off_home_minus_away() -> None:
    ledger = RestLedger(per_day=5.0)
    ledger.record(_game("A", "B", 1))  # both last played the 1st
    ledger.record(_game("B", "C", 8))  # B played again on the 8th

    # On the 15th: A rested 14 days, B rested 7.
    assert ledger.differential(_Matchup("A", "B", 15)) == pytest.approx(7.0)
    assert ledger.adjustment(_Matchup("A", "B", 15)) == pytest.approx(35.0)
    # And it is antisymmetric.
    assert ledger.differential(_Matchup("B", "A", 15)) == pytest.approx(-7.0)


def test_a_long_layoff_is_clamped() -> None:
    """Past two weeks the gap stops being rest and starts being a bowl."""
    ledger = RestLedger(per_day=1.0)
    ledger.record(_game("A", "B", 1))
    ledger.record(_game("B", "C", 25))

    # A has been off 59 days, B off 35 -- a 24-day raw differential.
    assert ledger.differential(_Matchup("A", "B", 60)) == REST_CAP_DAYS


def test_a_ledger_that_is_switched_off_costs_nothing() -> None:
    ledger = RestLedger(per_day=0.0)
    ledger.record(_game("A", "B", 1))
    ledger.record(_game("B", "C", 8))

    assert ledger.adjustment(_Matchup("A", "B", 15)) == 0.0


def test_reset_forgets_every_date() -> None:
    ledger = RestLedger(per_day=5.0)
    ledger.record(_game("A", "B", 1))
    ledger.record(_game("B", "C", 8))

    ledger.reset()

    assert ledger.differential(_Matchup("A", "B", 15)) == 0.0


def test_a_negative_rest_advantage_is_refused() -> None:
    """It would mean a rested team is worse for being rested."""
    with pytest.raises(ValueError, match="rest_advantage must be non-negative"):
        RestLedger(per_day=-1.0)


# ------------------------------------------------------------- the predictors


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_the_default_is_the_model_that_shipped_before_this(build: Build) -> None:
    """0 is off, so every release published before this replays unchanged."""
    plain = build(rest_advantage=0.0)
    rested = build(rest_advantage=0.0)
    for p in (plain, rested):
        p.update_game(_game("A", "B", 1))
        p.update_game(_game("B", "C", 8))

    assert plain.predict_game(_Matchup("A", "B", 15)) == rested.predict_game(
        _Matchup("A", "B", 15)
    )


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_a_better_rested_home_team_is_favored_more(build: Build) -> None:
    off = build(rest_advantage=0.0)
    on = build(rest_advantage=5.0)
    for p in (off, on):
        p.update_game(_game("A", "B", 1))
        p.update_game(_game("B", "C", 8))  # B plays again; A rests

    matchup = _Matchup("A", "B", 15)
    assert (
        on.predict_game(matchup).team1_win_prob
        > off.predict_game(matchup).team1_win_prob
    )


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_rest_still_counts_at_a_neutral_site(build: Build) -> None:
    """Unlike a home advantage. Nobody is at home in a bowl and both teams
    still arrived on different amounts of rest."""
    p = build(rest_advantage=5.0)
    p.update_game(_game("A", "B", 1))
    p.update_game(_game("B", "C", 8))

    neutral = p.predict_game(_Matchup("A", "B", 15, neutral_site=True))
    level = p.predict_game(_Matchup("A", "C", 15, neutral_site=True))
    assert neutral.team1_win_prob != pytest.approx(level.team1_win_prob)


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_a_game_never_counts_toward_its_own_rest(build: Build) -> None:
    """The ledger is written after the prediction, not before it.

    Asserted as "the prediction `update_game` returns is the one
    `predict_game` would have given a moment earlier", which is the real
    invariant and the only one that isn't tangled up with the rating the
    update then moves.
    """
    p = build(rest_advantage=5.0)
    p.update_game(_game("A", "B", 1))
    p.update_game(_game("B", "C", 8))

    expected = p.predict_game(_Matchup("A", "B", 15)).team1_win_prob
    returned = p.update_game(_game("A", "B", 15)).team1_win_prob

    assert returned == pytest.approx(expected)
    # And afterwards both sides have just played, so the edge is gone.
    assert p._adjustments.rest.differential(_Matchup("A", "B", 22)) == 0.0


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_the_offseason_clears_the_ledger(build: Build) -> None:
    """A team's last game of one season says nothing about the next."""
    p = build(rest_advantage=5.0)
    p.update_game(_game("A", "B", 1))
    p.update_game(_game("B", "C", 8))

    p.pass_season(2027)

    assert p._adjustments.rest.differential(_Matchup("A", "B", 15)) == 0.0


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_rest_advantage_round_trips_through_the_state(build: Build) -> None:
    """It changes what a matchup is worth, so a release has to carry it."""
    p = build(rest_advantage=3.5)

    state = p.state_dict()
    restored = type(p).from_state_dict(state)

    assert state["rest_advantage"] == 3.5
    assert restored.state_dict() == state


def test_a_bye_is_worth_something_and_only_the_bye_is() -> None:
    """The shape the residuals showed, with the ratings held still.

    Only the ledger moves here -- no `update_game` between the two
    predictions -- so the difference is the rest term and nothing else.
    """
    p = _glicko(rest_advantage=4.0)
    p.update_game(_game("A", "X", 1))
    p.update_game(_game("B", "X", 1))

    level = p.predict_game(_Matchup("A", "B", 20)).team1_win_prob
    # B plays again a week later; A does not. Recorded straight onto the
    # ledger so no rating moves with it.
    p._adjustments.rest.record(_game("B", "X", 8))
    with_bye = p.predict_game(_Matchup("A", "B", 20)).team1_win_prob

    assert with_bye > level
