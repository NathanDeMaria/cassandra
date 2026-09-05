"""Glicko rated on three signals at once.

The weighting itself is `blend_test`'s; what's here is the part specific to
this side of the experiment -- that control needs no conversion, that EPA gets
the scoring function's own logistic, and that the result is still a Glicko
update.
"""

import math

import pytest

from .conftest import GameFactory
from .epa import EpaIndex
from .game_control import GameControlIndex
from .glicko import GlickoPredictor
from .glicko_blend import DEFAULT_EPA_MARGIN_SCALE, BlendedGlickoPredictor
from .types import GameControl, GameEpa


def _predictor(
    control: dict[str, GameControl] | None = None,
    epa: dict[str, GameEpa] | None = None,
    *,
    play_weight: float = 0.5,
    epa_share: float = 0.5,
    epa_margin_scale: float = DEFAULT_EPA_MARGIN_SCALE,
) -> BlendedGlickoPredictor:
    return BlendedGlickoPredictor(
        "test_league",
        game_control=GameControlIndex(control or {}),
        game_epa=EpaIndex(epa or {}),
        play_weight=play_weight,
        epa_share=epa_share,
        epa_margin_scale=epa_margin_scale,
    )


def test_control_goes_in_without_conversion(game: GameFactory) -> None:
    """The one signal already on the target's scale.

    `cassandra.scoring` produces the home team's share of a game and so does
    the win probability curve, so 0.8 means 0.8.
    """
    predictor = _predictor(
        {"g1": GameControl(home=0.8, seconds=3600)}, play_weight=1.0, epa_share=0.0
    )

    assert predictor._actual(game("Home", "Away", 21, 17, game_id="g1")) == pytest.approx(
        0.8
    )


def test_epa_goes_through_the_scoring_functions_own_logistic(
    game: GameFactory,
) -> None:
    """A game EPA prices as a ten-point home win reads as a ten-point win.

    The conversion introduces no new opinion: it is `sigmoid_score`'s exact
    arithmetic, applied to a different set of points.
    """
    # 0.15 over 70 snaps minus 0.05 over 70 is a 7-point implied margin.
    epa = {"g1": GameEpa(home=0.15, away=0.05, home_plays=70, away_plays=70)}
    predictor = _predictor(epa=epa, play_weight=1.0, epa_share=1.0)

    assert predictor._actual(
        game("Home", "Away", 21, 17, game_id="g1")
    ) == pytest.approx(1 / (1 + math.exp(-7.0 / 10.0)))


def test_a_wider_margin_scale_softens_what_epa_claims(game: GameFactory) -> None:
    """The knob, and which direction it moves.

    A bigger divisor means more points are needed to say the same thing, so
    the same EPA reads as a closer game.
    """
    epa = {"g1": GameEpa(home=0.15, away=0.05, home_plays=70, away_plays=70)}
    played = game("Home", "Away", 21, 17, game_id="g1")

    tight = _predictor(epa=epa, play_weight=1.0, epa_share=1.0, epa_margin_scale=5.0)
    loose = _predictor(epa=epa, play_weight=1.0, epa_share=1.0, epa_margin_scale=20.0)

    assert tight._actual(played) > loose._actual(played) > 0.5


def test_the_epa_margin_scale_has_to_be_positive() -> None:
    with pytest.raises(ValueError, match="epa_margin_scale"):
        _predictor(epa_margin_scale=0.0)


def test_no_play_weight_is_glicko_exactly(game: GameFactory) -> None:
    """A search that finds the plays worthless recovers `GlickoPredictor`.

    Exactly, and with indexes full of numbers loud enough to show up if the
    short-circuit ever stopped short-circuiting.
    """
    played = game("Home", "Away", 31, 3, game_id="g1")
    control = {"g1": GameControl(home=0.02, seconds=3600)}
    epa = {"g1": GameEpa(home=-0.5, away=0.5, home_plays=70, away_plays=70)}

    blended = _predictor(control, epa, play_weight=0.0)
    plain = GlickoPredictor("test_league", scoring_method="sigmoid")
    blended.update_game(played)
    plain.update_game(played)

    assert blended.get_rating("Home") == plain.get_rating("Home")


def test_a_game_with_no_play_by_play_keeps_its_own_result(
    game: GameFactory,
) -> None:
    """Most of an NCAAFB schedule, at every weight including 1."""
    played = game("Home", "Away", 31, 3, game_id="unswept")

    for weight in (0.5, 1.0):
        blended = _predictor(play_weight=weight)
        plain = GlickoPredictor("test_league", scoring_method="sigmoid")
        blended.update_game(played)
        plain.update_game(played)
        assert blended.get_rating("Home") == plain.get_rating("Home")


def test_the_actual_stays_inside_the_unit_interval(game: GameFactory) -> None:
    """The one way this side is simpler than the margin side: nothing to
    clamp, because a convex combination of shares is a share."""
    control = {"g1": GameControl(home=0.99, seconds=3600)}
    epa = {"g1": GameEpa(home=3.0, away=-3.0, home_plays=70, away_plays=70)}
    predictor = _predictor(control, epa, play_weight=1.0)

    assert 0.0 <= predictor._actual(game("Home", "Away", 3, 31, game_id="g1")) <= 1.0


def test_a_controlled_game_rates_higher_than_a_stolen_one(
    game: GameFactory,
) -> None:
    """The point of the model, in one comparison.

    Two teams win by the same three points. One controlled the game and
    out-produced its opponent; the other was behind all afternoon and won it
    late. The blend rates the first higher, and plain Glicko -- which sees
    only the score -- rates them the same.
    """
    control = {
        "earned": GameControl(home=0.80, seconds=3600),
        "stolen": GameControl(home=0.30, seconds=3600),
    }
    epa = {
        "earned": GameEpa(home=0.25, away=-0.05, home_plays=70, away_plays=70),
        "stolen": GameEpa(home=-0.10, away=0.15, home_plays=70, away_plays=70),
    }

    ratings = []
    for game_id in ("earned", "stolen"):
        predictor = _predictor(control, epa, play_weight=0.5, epa_share=0.5)
        predictor.update_game(game("Home", "Away", 17, 14, game_id=game_id))
        ratings.append(predictor.get_rating("Home").rating)

    assert ratings[0] > ratings[1]

    plain = [GlickoPredictor("test_league", scoring_method="sigmoid") for _ in range(2)]
    for predictor, game_id in zip(plain, ("earned", "stolen")):
        predictor.update_game(game("Home", "Away", 17, 14, game_id=game_id))
    assert plain[0].get_rating("Home").rating == plain[1].get_rating("Home").rating


def test_the_away_side_gets_one_minus_the_blend(game: GameFactory) -> None:
    """Glicko updates the away team against `1 - actual`, so a blend that
    favours the home team has to move the two in opposite directions."""
    control = {"g1": GameControl(home=0.9, seconds=3600)}
    predictor = _predictor(control, play_weight=1.0, epa_share=0.0)

    predictor.update_game(game("Home", "Away", 17, 14, game_id="g1"))

    assert predictor.get_rating("Home").rating > 1500
    assert predictor.get_rating("Away").rating < 1500


def test_the_indexes_stay_out_of_the_state_dict() -> None:
    predictor = _predictor(
        {"g1": GameControl(home=0.6, seconds=3600)},
        {"g1": GameEpa(0.1, 0.0, 70, 70)},
        play_weight=0.4,
        epa_share=0.3,
    )

    state = predictor.state_dict()

    assert state["play_weight"] == pytest.approx(0.4)
    assert state["epa_share"] == pytest.approx(0.3)
    assert predictor.control_weight == pytest.approx(0.28)
    assert predictor.epa_weight == pytest.approx(0.12)
    assert "game_control" not in state
    assert "game_epa" not in state
