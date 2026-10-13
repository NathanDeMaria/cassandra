"""The margin model with offense and defense units on EPA per play.

The contract -- save, load, regress, anchor -- is `contract_test`'s. What's
here is the unit part: that each contest is a Kalman step on the offense and
the defense it faced, measured against the anchors and the league's center;
that each side keeps its own clock; that the smoother re-walks the units; and
that with the weight off or the index empty the parent is all there is.
"""

import json
import math
from collections.abc import Mapping
from typing import Any

import pytest

from ..offseason import OffseasonFact, OffseasonFacts
from ..roster import RosterIndex
from . import unit_glicko
from .base_predictor import MEAN_RATING, Anchor
from .conftest import GameFactory
from .epa import EpaIndex
from .margin_glicko import MarginGlickoPredictor
from .types import GameEpa, Rating
from .unit_glicko import (
    CENTER_PRIOR_SIDES,
    DEFAULT_DEFENSE_INITIAL_SD,
    DEFAULT_OFFENSE_INITIAL_SD,
    UNWEIGHTED,
    WEIGHTED,
    UnitMarginGlickoPredictor,
)


def _epa(home: float, away: float, plays: int = 70) -> GameEpa:
    """A game's EPA; the unit model reads only the flat pair and its snaps."""
    return GameEpa(home=home, away=away, home_plays=plays, away_plays=plays)


# A game where the home offense moved the ball and the away one didn't.
LOPSIDED = _epa(0.3, -0.3)


def _predictor(
    epa: Mapping[str, GameEpa] | None = None,
    anchors: Mapping[str, Anchor] | None = None,
    **params: Any,
) -> UnitMarginGlickoPredictor:
    return UnitMarginGlickoPredictor(
        "test_league",
        game_epa=EpaIndex(epa or {}),
        anchors=anchors if anchors is not None else {},
        **params,
    )


def _parents(predictor: UnitMarginGlickoPredictor) -> dict[str, Rating]:
    return {
        team: rating._replace(units=None, blended=None)
        for team, rating in predictor.ratings.items()
    }


@pytest.mark.parametrize("prediction_scale", [None, 170.0])
def test_unit_weight_zero_is_the_margin_model_game_by_game(
    game: GameFactory, prediction_scale: float | None
) -> None:
    """Every game has EPA, so the units move; they just may not speak."""
    epa = {str(i): LOPSIDED for i in range(4)}
    units = _predictor(epa, unit_weight=0.0, prediction_scale=prediction_scale)
    margin = MarginGlickoPredictor(
        "test_league", anchors={}, prediction_scale=prediction_scale
    )
    schedule = [
        game("A", "B", 21, 7, game_id="0"),
        game("C", "D", 10, 24, game_id="1"),
        game("A", "C", 14, 14, game_id="2"),
        game("D", "B", 3, 30, game_id="3"),
    ]
    for played in schedule:
        assert units.update_game(played) == margin.update_game(played)
    units.pass_week()
    margin.pass_week()

    assert units.unit_information("A") > 0
    assert _parents(units) == margin.ratings
    assert units.predict_game(game("B", "C")) == margin.predict_game(game("B", "C"))


def test_a_league_without_epa_is_the_margin_model_whatever_the_weight(
    game: GameFactory,
) -> None:
    units = _predictor(unit_weight=5.0)
    margin = MarginGlickoPredictor("test_league", anchors={})
    for played in (game("A", "B", 21, 7), game("B", "C", 3, 10)):
        assert units.update_game(played) == margin.update_game(played)

    assert units.unit_information("A") == 0
    assert units.ratings["A"].units is None
    assert units.predict_game(game("C", "A")) == margin.predict_game(game("C", "A"))


def test_the_offense_that_moved_the_ball_goes_up_and_the_defense_it_faced_goes_down(
    game: GameFactory,
) -> None:
    predictor = _predictor({"g": LOPSIDED})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    home, away = predictor.get_sides("A"), predictor.get_sides("B")
    # Offsets: a defense's rating is how much it holds offenses *below* par.
    assert home.offense.rating > 0 > away.defense.rating
    assert away.offense.rating < 0 < home.defense.rating
    for side in (*home, *away):
        assert side.rating_deviation < DEFAULT_OFFENSE_INITIAL_SD


def test_the_same_reading_against_a_better_defense_is_worth_more(
    game: GameFactory,
) -> None:
    """The opponent adjustment: a par day against an FBS defense is a good day."""
    even = _epa(0.0, 0.0)
    weak = _predictor({"g": even}, anchors={"B": 1200})
    strong = _predictor({"g": even}, anchors={"B": 1800})
    for predictor in (weak, strong):
        predictor.update_game(game("A", "B", 0, 0, game_id="g"))

    assert weak.get_sides("A").offense.rating < 0 < strong.get_sides("A").offense.rating


def test_the_anchor_scale_sets_how_much_of_the_anchor_gap_a_reading_owes(
    game: GameFactory,
) -> None:
    """Linear in the scale, and at 0 the anchors say nothing about a reading."""
    even = _epa(0.0, 0.0)
    worth = {}
    for scale in (0.0, 0.5, 1.0):
        offense = []
        for anchor in (1200, 1800):
            predictor = _predictor({"g": even}, anchors={"B": anchor}, anchor_epa_scale=scale)
            predictor.update_game(game("A", "B", 0, 0, game_id="g"))
            offense.append(predictor.get_sides("A").offense.rating)
        worth[scale] = offense[1] - offense[0]

    assert worth[0.0] == 0
    assert worth[1.0] > 0
    assert worth[0.5] == pytest.approx(worth[1.0] / 2)


def test_a_negative_anchor_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="anchor_epa_scale"):
        _predictor(anchor_epa_scale=-0.5)


def test_a_short_game_moves_the_units_less(game: GameFactory) -> None:
    short = _predictor({"g": _epa(0.3, -0.3, plays=20)})
    full = _predictor({"g": _epa(0.3, -0.3, plays=80)})
    for predictor in (short, full):
        predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    assert 0 < short.get_sides("A").offense.rating < full.get_sides("A").offense.rating


def test_the_center_is_the_average_offense_and_carries_into_the_next_season(
    game: GameFactory,
) -> None:
    predictor = _predictor({"g": _epa(0.2, 0.0), "h": _epa(0.0, 0.0)})
    assert predictor.epa_center == 0
    predictor.update_game(game("A", "B", 7, 7, game_id="g"))
    assert predictor.epa_center == pytest.approx(0.1)

    predictor.pass_season()
    # Last season's mean, alone, until this season has sides of its own...
    assert predictor.epa_center == pytest.approx(0.1)
    predictor.update_game(game("A", "B", 7, 7, game_id="h"))
    # ...and then pulled toward them by what they're worth against the prior.
    assert predictor.epa_center == pytest.approx(
        0.1 * CENTER_PRIOR_SIDES / (CENTER_PRIOR_SIDES + 2)
    )


def test_each_side_keeps_its_own_offseason(game: GameFactory) -> None:
    predictor = _predictor(
        {"g": LOPSIDED},
        offense_season_regression=0.0,
        defense_season_regression=1.0,
        offense_season_sd_increase=0.0,
        defense_season_sd_increase=1.0,
    )
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    before = predictor.get_sides("A")

    predictor.pass_season()

    after = predictor.get_sides("A")
    assert after.offense == before.offense
    assert after.defense.rating == 0
    # Grown back to its own cap, not past it.
    assert after.defense.rating_deviation == DEFAULT_DEFENSE_INITIAL_SD


def test_each_side_widens_by_its_own_weekly_increase(game: GameFactory) -> None:
    predictor = _predictor(
        {"g": LOPSIDED},
        offense_weekly_sd_increase=0.0,
        defense_weekly_sd_increase=0.05,
    )
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    before = predictor.get_sides("A")

    predictor.pass_week()

    after = predictor.get_sides("A")
    assert after.offense.rating_deviation == before.offense.rating_deviation
    assert after.defense.rating_deviation > before.defense.rating_deviation


def test_the_smoother_reprices_an_early_contest_and_keeps_the_deviations(
    game: GameFactory,
) -> None:
    """A week-1 par day against a defense week 2 exposed was a worse day."""
    epa = {"1": _epa(0.0, 0.0), "2": _epa(0.6, 0.0)}
    filtered = _predictor(epa, passes=1)
    smoothed = _predictor(epa, passes=3)
    for predictor in (filtered, smoothed):
        predictor.update_game(game("A", "B", 0, 0, game_id="1"))
        predictor.pass_week()
        predictor.update_game(game("C", "B", 0, 0, game_id="2"))
        predictor.pass_week()

    plain, walked = filtered.get_sides("A"), smoothed.get_sides("A")
    assert walked.offense.rating < plain.offense.rating
    assert walked.offense.rating_deviation == plain.offense.rating_deviation


def test_the_units_pull_the_prediction_toward_what_they_saw(
    game: GameFactory,
) -> None:
    """A 7-7 tie where A's offense moved the ball and B's didn't."""
    silent = _predictor({"g": LOPSIDED}, unit_weight=0.0)
    speaking = _predictor({"g": LOPSIDED}, unit_weight=1.0)
    for predictor in (silent, speaking):
        predictor.update_game(game("A", "B", 7, 7, game_id="g"))

    neutral = game("A", "B")._replace(neutral_site=True)
    assert (
        speaking.predict_game(neutral).team1_win_prob
        > silent.predict_game(neutral).team1_win_prob
    )


def test_a_team_is_ranked_by_the_rating_it_is_predicted_from(
    game: GameFactory,
) -> None:
    """A neutral-site tie leaves the parents level; only the blend can split them.

    So a table sorted on the parent calls them even while the model favors
    A, and `overall` -- what a release ranks on -- is the number that agrees
    with the prediction.
    """
    predictor = _predictor({"g": LOPSIDED}, unit_weight=1.0)
    predictor.update_game(game("A", "B", 7, 7, game_id="g")._replace(neutral_site=True))

    a, b = predictor.ratings["A"], predictor.ratings["B"]
    assert a.rating == pytest.approx(b.rating)
    assert a.overall == a.blended and a.overall > b.overall
    neutral = game("A", "B")._replace(neutral_site=True)
    assert predictor.predict_game(neutral).team1_win_prob > 0.5


def test_a_team_without_sides_is_blended_to_its_parent(game: GameFactory) -> None:
    predictor = _predictor(unit_weight=1.0)
    predictor.update_game(game("A", "B", 21, 7))

    rating = predictor.ratings["A"]
    assert rating.units is None
    assert rating.blended == pytest.approx(rating.rating)


def test_sides_go_out_on_the_team_scale_and_come_back(game: GameFactory) -> None:
    predictor = _predictor({"g": LOPSIDED}, anchors={"A": 1700, "B": 1300})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))

    released = predictor.ratings
    units = released["A"].units
    assert units is not None
    assert (units.offense.rating + units.defense.rating) / 2 == pytest.approx(
        predictor.unit_rating("A")
    )

    rebuilt = UnitMarginGlickoPredictor.from_ratings(
        "test_league",
        released,
        anchors={"A": 1700, "B": 1300},
        game_epa=EpaIndex(),
    )
    for team in ("A", "B"):
        for side, back in zip(predictor.get_sides(team), rebuilt.get_sides(team)):
            assert back.rating == pytest.approx(side.rating)
            assert back.rating_deviation == pytest.approx(side.rating_deviation)
    matchup = game("B", "A")
    assert rebuilt.predict_game(matchup).team1_win_prob == pytest.approx(
        predictor.predict_game(matchup).team1_win_prob
    )


def test_the_state_dict_carries_the_sides_and_the_center(game: GameFactory) -> None:
    predictor = _predictor({"g": LOPSIDED})
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    predictor.pass_season()

    state = json.loads(json.dumps(predictor.state_dict()))
    loaded = UnitMarginGlickoPredictor.from_state_dict(
        {**state, "game_epa": EpaIndex()}
    )

    assert loaded.get_sides("A") == predictor.get_sides("A")
    assert loaded.epa_center == predictor.epa_center
    assert loaded.predict_game(game("A", "B")) == predictor.predict_game(game("A", "B"))


def test_an_unseen_team_sits_at_its_anchor_by_its_units(game: GameFactory) -> None:
    predictor = _predictor(anchors={"A": 1200})
    assert predictor.unit_rating("A") == 1200
    assert predictor.unit_rating("B") == MEAN_RATING


@pytest.mark.parametrize(
    "param",
    [
        {"unit_weight": -0.1},
        {"points_per_epa": 0.0},
        {"play_sd": -1.0},
        {"offense_initial_sd": 0.0},
        {"defense_weekly_sd_increase": -0.01},
        {"offense_season_sd_increase": -0.01},
        {"defense_season_regression": 1.5},
        {"talent_shift": -0.5},
        {"talent_edge": -0.5},
        {"offense_returning_shift": -0.5},
        {"defense_returning_shift": -0.5},
    ],
)
def test_nonsense_unit_settings_are_refused(param: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _predictor(**param)


# --- offseason shifts ---------------------------------------------------------


COACH_LEFT = OffseasonFacts({("A", 2024): OffseasonFact("left_for_job", None, 0.0)})
# A new starter a tenth of a point per attempt better than the one who left.
BETTER_QB = OffseasonFacts({("A", 2024): OffseasonFact(None, True, 0.1)})


def _into_2024(predictor: UnitMarginGlickoPredictor, game: GameFactory) -> None:
    predictor.pass_season(2023)
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    predictor.pass_week()
    predictor.pass_season(2024)


def test_no_shift_knobs_is_the_model_without_facts(game: GameFactory) -> None:
    facts = OffseasonFacts({("A", 2024): OffseasonFact("left_for_job", True, 0.2)})
    plain = _predictor({"g": LOPSIDED})
    read = _predictor({"g": LOPSIDED}, offseason=facts)
    for predictor in (plain, read):
        _into_2024(predictor, game)
        predictor.update_game(game("A", "C", 3, 10, game_id="h"))
    assert read.ratings == plain.ratings
    assert read.predict_game(game("A", "B")) == plain.predict_game(game("A", "B"))


def test_a_coach_who_left_moves_the_team_and_its_offense_at_the_rollover(
    game: GameFactory,
) -> None:
    plain = _predictor({"g": LOPSIDED}, coach_left_shift=-3.0)
    left = _predictor({"g": LOPSIDED}, coach_left_shift=-3.0, offseason=COACH_LEFT)
    for predictor in (plain, left):
        _into_2024(predictor, game)

    moved = left.get_rating("A").rating - plain.get_rating("A").rating
    assert moved == pytest.approx(-3.0 / left.points_per_rating)
    offense = left.get_sides("A").offense.rating - plain.get_sides("A").offense.rating
    assert offense == pytest.approx(-3.0 / left.points_per_epa)
    assert left.get_rating("B") == plain.get_rating("B")


def test_a_new_quarterback_waits_for_his_first_game(game: GameFactory) -> None:
    # -1 + 2 * 1 = +1 point
    knobs: dict[str, Any] = {"new_qb_shift": -1.0, "qb_quality_shift": 2.0}
    plays = {"g": LOPSIDED, "h": LOPSIDED, "i": LOPSIDED}
    plain = _predictor(plays, **knobs)
    new = _predictor(plays, offseason=BETTER_QB, **knobs)
    for predictor in (plain, new):
        _into_2024(predictor, game)

    # Nothing is known about him until he has played.
    assert new.get_rating("A") == plain.get_rating("A")
    opener = game("A", "C", 14, 10, game_id="h")
    assert new.update_game(opener) == plain.update_game(opener)

    moved = new.get_rating("A").rating - plain.get_rating("A").rating
    assert moved == pytest.approx(1.0 / new.points_per_rating)
    # Once, not every game.
    second = game("C", "A", 7, 7, game_id="i")
    new.update_game(second)
    plain.update_game(second)
    again = new.get_rating("A").rating - plain.get_rating("A").rating
    assert again < moved


def test_a_quarterback_shift_survives_the_smoother(game: GameFactory) -> None:
    knobs: dict[str, Any] = {"new_qb_shift": 1.0, "passes": 3}
    plain = _predictor({"g": LOPSIDED, "h": LOPSIDED}, **knobs)
    new = _predictor({"g": LOPSIDED, "h": LOPSIDED}, offseason=BETTER_QB, **knobs)
    for predictor in (plain, new):
        _into_2024(predictor, game)
        predictor.update_game(game("A", "C", 14, 10, game_id="h"))
        predictor.pass_week()

    moved = new.get_rating("A").rating - plain.get_rating("A").rating
    assert moved > 0.5 / new.points_per_rating


def test_the_shift_knobs_round_trip(game: GameFactory) -> None:
    predictor = _predictor(
        coach_left_shift=-2.5,
        new_qb_shift=-1.1,
        qb_quality_shift=1.7,
        talent_shift=0.8,
        talent_edge=1.2,
        offense_returning_shift=0.9,
        defense_returning_shift=0.7,
    )
    state = json.loads(json.dumps(predictor.state_dict()))
    loaded = UnitMarginGlickoPredictor.from_state_dict(
        {
            **state,
            "game_epa": EpaIndex(),
            "offseason": OffseasonFacts(),
            "talent": RosterIndex(),
            "offense_returning": RosterIndex(),
            "defense_returning": RosterIndex(),
        }
    )
    assert loaded.state_dict() == predictor.state_dict()


# A's roster 1.5 sd above the season's FBS mean.
TALENTED = RosterIndex({("A", 2024): 1.5})


def test_no_talent_shift_is_the_model_without_talent(game: GameFactory) -> None:
    plain = _predictor({"g": LOPSIDED})
    read = _predictor({"g": LOPSIDED}, talent=TALENTED)
    for predictor in (plain, read):
        _into_2024(predictor, game)
    assert read.ratings == plain.ratings


def test_talent_moves_the_team_and_its_offense_at_the_rollover(
    game: GameFactory,
) -> None:
    plain = _predictor({"g": LOPSIDED}, talent_shift=0.8)
    read = _predictor({"g": LOPSIDED}, talent_shift=0.8, talent=TALENTED)
    for predictor in (plain, read):
        _into_2024(predictor, game)

    moved = read.get_rating("A").rating - plain.get_rating("A").rating
    assert moved == pytest.approx(0.8 * 1.5 / read.points_per_rating)
    offense = read.get_sides("A").offense.rating - plain.get_sides("A").offense.rating
    assert offense == pytest.approx(0.8 * 1.5 / read.points_per_epa)
    assert read.get_rating("B") == plain.get_rating("B")


def test_an_opener_without_plays_does_not_name_the_starter(game: GameFactory) -> None:
    """The starter on file is the next game's, so the shift waits for it."""
    knobs: dict[str, Any] = {"new_qb_shift": 1.0}
    plain = _predictor({"g": LOPSIDED, "i": LOPSIDED}, **knobs)
    new = _predictor({"g": LOPSIDED, "i": LOPSIDED}, offseason=BETTER_QB, **knobs)
    for predictor in (plain, new):
        _into_2024(predictor, game)
        predictor.update_game(game("A", "C", 35, 3, game_id="h"))  # no plays

    assert new.get_rating("A") == plain.get_rating("A")
    second = game("A", "D", 21, 17, game_id="i")
    assert new.update_game(second) == plain.update_game(second)
    assert new.get_rating("A").rating > plain.get_rating("A").rating


def test_each_summer_shifts_only_the_change_in_talent(game: GameFactory) -> None:
    # 1.0 sd in 2024 and 1.5 in 2025: by 2025 the shifts add up to 1.5 sd's
    # worth, not 2.5.
    talent = RosterIndex({("A", 2024): 1.0, ("A", 2025): 1.5})
    plain = _predictor({"g": LOPSIDED}, talent_shift=0.8)
    read = _predictor({"g": LOPSIDED}, talent_shift=0.8, talent=talent)
    for predictor in (plain, read):
        _into_2024(predictor, game)
        predictor.pass_season(2025)

    moved = read.get_rating("A").rating - plain.get_rating("A").rating
    assert moved == pytest.approx(0.8 * 1.5 / read.points_per_rating)


def test_the_talent_gap_is_a_matchup_term(game: GameFactory) -> None:
    # A 1.5 sd above the FBS mean, B 0.5 below: a 2 sd gap, 2 points at
    # 1 point per sd.
    talent = RosterIndex({("A", 2024): 1.5, ("B", 2024): -0.5})
    plain = _predictor({"g": LOPSIDED}, talent_edge=1.0)
    read = _predictor({"g": LOPSIDED}, talent_edge=1.0, talent=talent)
    for predictor in (plain, read):
        _into_2024(predictor, game)

    assert read.ratings == plain.ratings
    gap = read.matchup_adjustment(game("A", "B")) - plain.matchup_adjustment(
        game("A", "B")
    )
    assert gap == pytest.approx(2.0 / read.points_per_rating)
    assert read.matchup_adjustment(game("B", "A")) == pytest.approx(
        plain.matchup_adjustment(game("B", "A")) - gap
    )
    assert (
        read.predict_game(game("A", "B")).team1_win_prob
        > plain.predict_game(game("A", "B")).team1_win_prob
    )


def test_an_unrated_side_prices_no_talent(game: GameFactory) -> None:
    talent = RosterIndex({("A", 2024): 1.5})
    plain = _predictor({"g": LOPSIDED}, talent_edge=1.0)
    read = _predictor({"g": LOPSIDED}, talent_edge=1.0, talent=talent)
    for predictor in (plain, read):
        _into_2024(predictor, game)
    assert read.predict_game(game("A", "C")) == plain.predict_game(game("A", "C"))


@pytest.mark.parametrize("side", ["offense", "defense"])
def test_returning_production_shifts_the_rating_every_summer(
    game: GameFactory, side: str
) -> None:
    # A returns 1.5 sd above the FBS mean in 2024 and 0.5 in 2025: each
    # summer shifts by that summer's level, which the games in between price.
    returning = RosterIndex({("A", 2024): 1.5, ("A", 2025): 0.5})
    knob: dict[str, Any] = {f"{side}_returning_shift": 0.8}
    index: dict[str, Any] = {f"{side}_returning": returning}
    plain = _predictor({"g": LOPSIDED}, **knob)
    read = _predictor({"g": LOPSIDED}, **knob, **index)
    for predictor in (plain, read):
        _into_2024(predictor, game)

    moved = read.get_rating("A").rating - plain.get_rating("A").rating
    assert moved == pytest.approx(0.8 * 1.5 / read.points_per_rating)
    # The units are learned from plays and don't hear about the roster.
    assert read.get_sides("A") == plain.get_sides("A")
    assert read.get_rating("B") == plain.get_rating("B")

    for predictor in (plain, read):
        predictor.pass_season(2025)
    moved = read.get_rating("A").rating - plain.get_rating("A").rating
    assert moved == pytest.approx(0.8 * (1.5 + 0.5) / read.points_per_rating)


def test_returning_production_is_not_a_matchup_term(game: GameFactory) -> None:
    returning = RosterIndex({("A", 2024): 1.0, ("B", 2024): -1.0})
    plain = _predictor({"g": LOPSIDED})
    read = _predictor(
        {"g": LOPSIDED}, offense_returning=returning, defense_returning=returning
    )
    for predictor in (plain, read):
        _into_2024(predictor, game)
    assert read.matchup_adjustment(game("A", "B")) == plain.matchup_adjustment(
        game("A", "B")
    )


# --- success rate -------------------------------------------------------------


# The test league's success rate on EPA's scale: 1.5 x rate - 0.6, so a 60%
# day reads as +0.3 and a 40% day as 0.0. The weighted rate on its own,
# which puts 40% at 0.0 too.
_SCALE = (1.5, -0.6)
_WEIGHTED_SCALE = (1.0, -0.4)


@pytest.fixture
def scaled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        unit_glicko.SUCCESS_SCALE,
        "test_league",
        {UNWEIGHTED: _SCALE, WEIGHTED: _WEIGHTED_SCALE},
    )


def _both(epa: float, success: float, plays: int = 70) -> GameEpa:
    """A game the home offense read `epa` and `success` in; the away side read par."""
    return GameEpa(epa, 0.0, plays, plays, None, None, 0.0, 0.0, success, 0.4, plays, plays)


@pytest.mark.usefixtures("scaled")
def test_no_success_weight_is_the_epa_model(game: GameFactory) -> None:
    plain = _predictor({"g": LOPSIDED})
    carried = _predictor({"g": LOPSIDED._replace(home_success=0.2, away_success=0.7)})
    for predictor in (plain, carried):
        predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    assert carried.get_sides("A") == plain.get_sides("A")
    assert carried.get_sides("B") == plain.get_sides("B")


@pytest.mark.usefixtures("scaled")
def test_a_full_weight_reads_the_success_rate_and_not_the_epa(game: GameFactory) -> None:
    """A game that lost EPA on one long turnover and won most of its downs."""
    predictor = _predictor({"g": _both(-0.2, 0.6)}, success_weight=1.0)
    predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    assert predictor.get_sides("A").offense.rating > 0


@pytest.mark.usefixtures("scaled")
def test_the_blend_sits_between_its_two_readings(game: GameFactory) -> None:
    """Linear in the weight: the filter's step is linear in what it reads."""
    offense = {}
    for weight in (0.0, 0.5, 1.0):
        predictor = _predictor({"g": _both(-0.2, 0.6)}, success_weight=weight)
        predictor.update_game(game("A", "B", 21, 7, game_id="g"))
        offense[weight] = predictor.get_sides("A").offense.rating
    assert offense[0.5] == pytest.approx((offense[0.0] + offense[1.0]) / 2)


@pytest.mark.usefixtures("scaled")
def test_a_game_without_a_success_rate_reads_its_epa(game: GameFactory) -> None:
    plain = _predictor({"g": LOPSIDED})
    blended = _predictor({"g": LOPSIDED}, success_weight=0.75)
    for predictor in (plain, blended):
        predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    assert blended.get_sides("A") == plain.get_sides("A")


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_a_success_weight_outside_the_blend_is_refused(weight: float) -> None:
    with pytest.raises(ValueError):
        _predictor(success_weight=weight)


def test_a_success_weight_needs_the_league_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        _predictor(success_weight=0.5)


@pytest.mark.usefixtures("scaled")
def test_the_weighted_reading_reads_the_weighted_rate(game: GameFactory) -> None:
    """A rout won on its backups' downs: 60% flat, 30% while the game was live."""
    rout = _both(0.0, 0.6)._replace(home_success_weighted=0.3, away_success_weighted=0.4)
    flat = _predictor({"g": rout}, success_weight=1.0)
    weighted = _predictor({"g": rout}, success_weight=1.0, success_reading=WEIGHTED)
    for predictor in (flat, weighted):
        predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    assert flat.get_sides("A").offense.rating > 0
    assert weighted.get_sides("A").offense.rating < 0


@pytest.mark.usefixtures("scaled")
def test_a_game_without_a_weighted_rate_reads_its_epa(game: GameFactory) -> None:
    """An index swept before the weighted rate, or a side that only ever ran out the clock."""
    plain = _predictor({"g": _both(0.1, 0.6)})
    weighted = _predictor(
        {"g": _both(0.1, 0.6)}, success_weight=0.75, success_reading=WEIGHTED
    )
    for predictor in (plain, weighted):
        predictor.update_game(game("A", "B", 21, 7, game_id="g"))
    assert weighted.get_sides("A") == plain.get_sides("A")


def test_an_unknown_success_reading_is_refused() -> None:
    with pytest.raises(ValueError, match="success_reading"):
        _predictor(success_reading="garbage_time")


# --- a new head coach ---------------------------------------------------------


FIRED = OffseasonFacts({("A", 2024): OffseasonFact("fired", None, 0.0)})


def test_a_new_coach_widens_the_record_and_each_side_by_its_own(
    game: GameFactory,
) -> None:
    knobs: dict[str, Any] = {
        "coach_change_rd": 40.0,
        "coach_change_offense_sd": 0.2,
        "coach_change_defense_sd": 0.0,
    }
    plain = _predictor({"g": LOPSIDED}, **knobs)
    fired = _predictor({"g": LOPSIDED}, offseason=FIRED, **knobs)
    for predictor in (plain, fired):
        _into_2024(predictor, game)

    before, after = plain.get_rating("A"), fired.get_rating("A")
    assert after.rating == before.rating
    assert after.rating_deviation == pytest.approx(
        min(fired._initial_rd, math.hypot(before.rating_deviation, 40.0))
    )
    old, new = plain.get_sides("A"), fired.get_sides("A")
    # Past the side's own cap: a new staff is more than a season's forgetting.
    assert new.offense.rating_deviation == pytest.approx(
        math.hypot(old.offense.rating_deviation, 0.2)
    )
    assert new.offense.rating_deviation > DEFAULT_OFFENSE_INITIAL_SD
    assert new.defense == old.defense
    assert fired.get_sides("B") == plain.get_sides("B")


def test_a_widened_side_is_not_narrowed_by_a_quiet_week(game: GameFactory) -> None:
    predictor = _predictor(
        {"g": LOPSIDED},
        offseason=FIRED,
        coach_change_offense_sd=0.2,
        offense_weekly_sd_increase=0.05,
    )
    _into_2024(predictor, game)
    widened = predictor.get_sides("A").offense.rating_deviation

    predictor.pass_week()

    assert predictor.get_sides("A").offense.rating_deviation == widened


@pytest.mark.usefixtures("scaled")
def test_the_new_knobs_round_trip() -> None:
    predictor = _predictor(
        success_weight=0.6,
        success_reading=WEIGHTED,
        anchor_epa_scale=0.8,
        coach_change_rd=35.0,
        coach_change_offense_sd=0.1,
        coach_change_defense_sd=0.05,
    )
    state = json.loads(json.dumps(predictor.state_dict()))
    loaded = UnitMarginGlickoPredictor.from_state_dict(
        {
            **state,
            "game_epa": EpaIndex(),
            "offseason": OffseasonFacts(),
            "talent": RosterIndex(),
            "offense_returning": RosterIndex(),
            "defense_returning": RosterIndex(),
        }
    )
    assert loaded.state_dict() == predictor.state_dict()
