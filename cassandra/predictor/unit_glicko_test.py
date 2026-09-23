"""The margin model with offense and defense units on EPA per play.

The contract -- save, load, regress, anchor -- is `contract_test`'s. What's
here is the unit part: that each contest is a Kalman step on the offense and
the defense it faced, measured against the anchors and the league's center;
that each side keeps its own clock; that the smoother re-walks the units; and
that with the weight off or the index empty the parent is all there is.
"""

import json
from collections.abc import Mapping
from typing import Any

import pytest

from .base_predictor import MEAN_RATING, Anchor
from .conftest import GameFactory
from .epa import EpaIndex
from .margin_glicko import MarginGlickoPredictor
from .types import GameEpa, Rating
from .unit_glicko import (
    CENTER_PRIOR_SIDES,
    DEFAULT_DEFENSE_INITIAL_SD,
    DEFAULT_OFFENSE_INITIAL_SD,
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
        team: rating._replace(units=None) for team, rating in predictor.ratings.items()
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
    ],
)
def test_nonsense_unit_settings_are_refused(param: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _predictor(**param)
