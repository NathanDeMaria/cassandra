"""Tests for the rest advantage: the ledger, and what it does to a prediction."""

from collections.abc import Callable
from datetime import datetime, timedelta

import pytest
from endgame.types import Game

from .adjustments import MatchupAdjustments, MatchupSources
from .base_predictor import Predictor
from .elo import EloPredictor
from .glicko import GlickoPredictor
from .opponent_prior import OpponentPriorManager
from .rest import (
    REST_MAX_GAP_DAYS,
    REST_THRESHOLD_DAYS,
    RestLedger,
    RestSource,
    StatedRest,
)

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


def _priced(rest: RestSource, points: float) -> MatchupAdjustments:
    """A rest source with a weight on it, which is what a predictor holds."""
    return MatchupAdjustments(
        rest_advantage=points, sources=MatchupSources.empty()._replace(rest=rest)
    )


def _ledger(predictor: Predictor) -> RestLedger:
    """The walking ledger a predictor built for itself.

    For the tests that want the raw day count: `differential` is the ledger's
    own diagnostic and deliberately not on `RestSource`, because a source that
    was told who is rested has no number of days to report.
    """
    rest = predictor._adjustments.sources.rest
    assert isinstance(rest, RestLedger)
    return rest


# ---------------------------------------------------------------- the ledger


def test_a_season_opener_has_no_differential() -> None:
    """ "No idea" and "extremely rested" are different claims."""
    ledger = RestLedger()
    ledger.record(_game("A", "B", 1))

    # B has a date, C never played -- so there is nothing to compare.
    assert ledger.differential(_Matchup("B", "C", 8)) == 0.0
    assert ledger.rested_side(_Matchup("B", "C", 8)) == 0.0


def test_the_differential_is_days_off_home_minus_away() -> None:
    ledger = RestLedger()
    ledger.record(_game("A", "B", 1))  # both last played the 1st
    ledger.record(_game("B", "C", 8))  # B played again on the 8th

    # On the 15th: A rested 14 days, B rested 7.
    assert ledger.differential(_Matchup("A", "B", 15)) == pytest.approx(7.0)
    # And it is antisymmetric.
    assert ledger.differential(_Matchup("B", "A", 15)) == pytest.approx(-7.0)


def test_the_bump_is_flat_once_the_threshold_is_cleared() -> None:
    """A bye is a bye. Thirteen days better rested is not twice seven."""
    seven = RestLedger()
    seven.record(_game("A", "B", 1))
    seven.record(_game("B", "C", 8))  # A off 14, B off 7 on day 15

    thirteen = RestLedger()
    thirteen.record(_game("A", "B", 1))
    thirteen.record(_game("B", "C", 14))  # A off 14, B off 1 on day 15

    assert seven.rested_side(_Matchup("A", "B", 15)) == 1.0
    assert thirteen.rested_side(_Matchup("A", "B", 15)) == 1.0
    assert seven.rested_side(_Matchup("B", "A", 15)) == -1.0


def test_a_gap_under_the_threshold_is_worth_nothing() -> None:
    """A Thursday game is a different fact from a bye, and measured it does
    not run the same way."""
    ledger = RestLedger()
    ledger.record(_game("A", "B", 1))
    ledger.record(_game("B", "C", 4))  # B played 3 days later

    # A is only 3 days better rested, under the 5-day threshold.
    assert ledger.differential(_Matchup("A", "B", 8)) == pytest.approx(3.0)
    assert ledger.rested_side(_Matchup("A", "B", 8)) == 0.0


def test_a_gap_too_long_to_be_a_bye_is_read_as_missing_data() -> None:
    """A team is not off for three weeks mid-season. Far more often the game
    it played is a row the season file doesn't have, and reading the hole as
    a bye would invent that team's largest adjustment of the year."""
    ledger = RestLedger()
    ledger.record(_game("A", "B", 1))
    ledger.record(_game("B", "C", 8))

    # A has been off 24 days by day 25 -- past the guard.
    assert ledger.differential(_Matchup("A", "B", 25)) == pytest.approx(7.0)
    assert ledger.rested_side(_Matchup("A", "B", 25)) == 0.0


def test_one_suspect_side_is_enough_to_throw_the_comparison_out() -> None:
    """The differential is a difference, and only as good as its worse half.

    Here the *away* side is the one with the hole, and the home side looks
    like an ordinary week -- the adjustment still has to stand down.
    """
    ledger = RestLedger()
    ledger.record(_game("B", "C", 1))  # B last played on day 1
    ledger.record(_game("A", "C", 18))  # A last played on day 18

    # On day 25: A off 7 (normal), B off 24 (a hole).
    assert ledger.rested_side(_Matchup("A", "B", 25)) == 0.0


def test_the_guard_sits_above_every_ordinary_schedule() -> None:
    """A real bye is 14 days and two in a row is 21."""
    assert REST_THRESHOLD_DAYS == 5.0
    assert REST_MAX_GAP_DAYS == 20.0


def test_a_weight_of_zero_costs_nothing() -> None:
    """The off switch, which is what every model before this replayed with.

    Asserted on the adjustment rather than the ledger: the ledger still says
    who is rested, and the weight is the claim about what that is worth.
    """
    ledger = RestLedger()
    ledger.record(_game("A", "B", 1))
    ledger.record(_game("B", "C", 8))
    matchup = _Matchup("A", "B", 15)

    assert ledger.rested_side(matchup) == 1.0
    assert _priced(ledger, 0.0).rest_points(matchup) == 0.0
    assert _priced(ledger, 5.0).rest_points(matchup) == pytest.approx(5.0)


def test_reset_forgets_every_date() -> None:
    ledger = RestLedger()
    ledger.record(_game("A", "B", 1))
    ledger.record(_game("B", "C", 8))

    ledger.reset()

    assert ledger.differential(_Matchup("A", "B", 15)) == 0.0


def test_a_negative_rest_advantage_is_refused() -> None:
    """It would mean a rested team is worse for being rested."""
    with pytest.raises(ValueError, match="rest_advantage must be non-negative"):
        MatchupAdjustments(rest_advantage=-1.0)


# ------------------------------------------------------------------ stated rest


def test_a_stated_rest_names_the_rested_side() -> None:
    """The fixture case: no dates to walk, so a caller says who is off a bye."""
    stated = StatedRest({"m": "A"})

    assert stated.rested_side(_Matchup("A", "B", 15)) == 1.0
    assert stated.rested_side(_Matchup("B", "A", 15)) == -1.0


def test_a_game_nobody_stated_has_nobody_rested() -> None:
    """Same answer an empty ledger gives, and the same one it should be."""
    assert StatedRest().rested_side(_Matchup("A", "B", 15)) == 0.0
    assert StatedRest({"other": "A"}).rested_side(_Matchup("A", "B", 15)) == 0.0


def test_a_stated_team_that_is_not_playing_is_ignored() -> None:
    """A typo or a stale fixture, and reading it as "the other side, then"
    would turn a bad input into the largest adjustment on the board."""
    assert StatedRest({"m": "Somebody Else"}).rested_side(_Matchup("A", "B", 15)) == 0.0


def test_a_stated_rest_has_nothing_to_learn_or_forget() -> None:
    """It was told its answer, so the replay's calls are no-ops on it."""
    stated = StatedRest({"m": "A"})
    matchup = _Matchup("A", "B", 15)

    stated.record(_game("A", "B", 1))
    stated.reset()

    assert stated.rested_side(matchup) == 1.0


def test_a_stated_rest_prices_like_a_walked_one() -> None:
    """The point of the seam: the predictor cannot tell which it was handed."""
    walked = RestLedger()
    walked.record(_game("A", "B", 1))
    walked.record(_game("B", "C", 8))
    matchup = _Matchup("A", "B", 15)

    assert _priced(walked, 5.0).rest_points(matchup) == pytest.approx(
        _priced(StatedRest({"m": "A"}), 5.0).rest_points(matchup)
    )


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
    assert _ledger(p).differential(_Matchup("A", "B", 22)) == 0.0


@pytest.mark.parametrize("build", [_elo, _glicko], ids=["elo", "glicko"])
def test_the_offseason_clears_the_ledger(build: Build) -> None:
    """A team's last game of one season says nothing about the next."""
    p = build(rest_advantage=5.0)
    p.update_game(_game("A", "B", 1))
    p.update_game(_game("B", "C", 8))

    p.pass_season(2027)

    assert _ledger(p).differential(_Matchup("A", "B", 15)) == 0.0


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
    _ledger(p).record(_game("B", "X", 8))
    with_bye = p.predict_game(_Matchup("A", "B", 20)).team1_win_prob

    assert with_bye > level
