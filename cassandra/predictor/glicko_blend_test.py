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
from .glicko_blend import (
    DEFAULT_CONTROL_TEMP,
    DEFAULT_EPA_MARGIN_SCALE,
    DEFAULT_EPA_RESIDUAL_BETA,
    DEFAULT_MOV_SCALE,
    BlendedGlickoPredictor,
)
from .types import GameControl, GameEpa


def _predictor(
    control: dict[str, GameControl] | None = None,
    epa: dict[str, GameEpa] | None = None,
    *,
    play_weight: float = 0.5,
    epa_share: float = 0.5,
    epa_margin_scale: float = DEFAULT_EPA_MARGIN_SCALE,
    mov_scale: float = DEFAULT_MOV_SCALE,
    control_temp: float = DEFAULT_CONTROL_TEMP,
    epa_residual_beta: float = DEFAULT_EPA_RESIDUAL_BETA,
) -> BlendedGlickoPredictor:
    return BlendedGlickoPredictor(
        "test_league",
        game_control=GameControlIndex(control or {}),
        game_epa=EpaIndex(epa or {}),
        play_weight=play_weight,
        epa_share=epa_share,
        epa_margin_scale=epa_margin_scale,
        mov_scale=mov_scale,
        control_temp=control_temp,
        epa_residual_beta=epa_residual_beta,
    )


def test_control_at_temperature_one_is_untouched(game: GameFactory) -> None:
    """The default leaves the index's own number alone, so 0.8 means 0.8.

    Control is the one source already on the target's scale, so its identity
    temperature is a passthrough rather than a divisor that happens to cancel.
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


def test_the_defaults_reproduce_the_scoring_function_they_replaced(
    game: GameFactory,
) -> None:
    """`mov_scale` at 10 is `sigmoid_score` exactly.

    The parent's scorer is gone from this class, so this is what keeps the
    replacement honest: every release published by the version that used
    `scoring_method="sigmoid"` replays through the new path unchanged.
    """
    played = game("Home", "Away", 27, 13, game_id="g1")

    blended = _predictor(play_weight=0.0)
    plain = GlickoPredictor("test_league", scoring_method="sigmoid")

    assert blended._actual(played) == pytest.approx(plain._actual(played))


@pytest.mark.parametrize("scale", [1.0, 10.0, 30.0])
def test_the_mov_temperature_decides_how_much_a_blowout_counts(
    scale: float, game: GameFactory
) -> None:
    """What the temperature is for, in one comparison.

    A close win and a blowout are the same event to a hard scorer and
    different events to a soft one, so the gap between them has to shrink as
    the scale widens.
    """
    close = _predictor(play_weight=0.0, mov_scale=scale)._actual(
        game("Home", "Away", 20, 17, game_id="g1")
    )
    blowout = _predictor(play_weight=0.0, mov_scale=scale)._actual(
        game("Home", "Away", 49, 3, game_id="g2")
    )

    assert 0.5 < close < blowout <= 1.0


def test_a_small_mov_scale_approaches_binary_scoring(game: GameFactory) -> None:
    """`binary_score` is the limit, which is why dropping `scoring_method`
    loses nothing: a categorical became an end of a continuum."""
    predictor = _predictor(play_weight=0.0, mov_scale=0.01)

    assert predictor._actual(game("Home", "Away", 20, 17, game_id="g1")) == pytest.approx(
        1.0
    )
    assert predictor._actual(game("Home", "Away", 17, 20, game_id="g2")) == pytest.approx(
        0.0
    )


def test_the_control_temperature_sharpens_and_flattens_around_a_half() -> None:
    """Below 1 stretches control away from 0.5, above 1 pushes it back.

    Control is compressed toward 0.5 by construction -- every game starts at
    even odds -- so this is the knob that says whether that compression is
    information or an artefact.
    """
    control = {"g1": GameControl(home=0.8, seconds=3600)}

    sharp = _predictor(control, control_temp=0.5)._control_share_of_game("g1")
    flat = _predictor(control, control_temp=2.0)._control_share_of_game("g1")
    plain = _predictor(control, control_temp=1.0)._control_share_of_game("g1")

    assert sharp is not None and flat is not None and plain is not None
    assert sharp > plain > flat > 0.5


def test_the_control_temperature_leaves_an_even_game_even() -> None:
    """0.5 is the fixed point of the transform, at every temperature: a game
    nobody controlled says nothing however hard you listen."""
    control = {"g1": GameControl(home=0.5, seconds=3600)}

    for temp in (0.2, 1.0, 5.0):
        assert _predictor(control, control_temp=temp)._control_share_of_game(
            "g1"
        ) == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["mov_scale", "control_temp"])
def test_a_temperature_has_to_be_positive(name: str, game: GameFactory) -> None:
    """0 is not the binary limit, it is a division by zero, and negative
    reads every game backwards."""
    with pytest.raises(ValueError, match=name):
        _predictor(**{name: 0.0})  # ty: ignore[invalid-argument-type]


def test_the_temperatures_round_trip_through_a_release() -> None:
    """They decide what every game was worth, so a release without them
    replays as a different model."""
    predictor = _predictor(mov_scale=7.5, control_temp=0.4, epa_margin_scale=21.0)

    state = predictor.state_dict()
    restored = BlendedGlickoPredictor.from_state_dict(state)

    assert state["mov_scale"] == 7.5
    assert state["control_temp"] == 0.4
    assert state["epa_margin_scale"] == 21.0
    assert restored.state_dict() == state
    # The parent's scorer is not part of this model any more.
    assert "scoring_method" not in state


def test_the_residual_is_off_by_default(game: GameFactory) -> None:
    """`epa_residual_beta` at 0 is the EPA margin whole, which is what this
    class did before it could subtract anything."""
    epa = {"g1": GameEpa(home=0.15, away=0.05, home_plays=70, away_plays=70)}
    played = game("Home", "Away", 31, 10, game_id="g1")

    predictor = _predictor(epa=epa, play_weight=1.0, epa_share=1.0)

    assert predictor._epa_share_of_game(played) == pytest.approx(
        1 / (1 + math.exp(-7.0 / 10.0))
    )


def test_the_residual_subtracts_the_scoreboard(game: GameFactory) -> None:
    """7 points of EPA margin in a game won by 21, at beta 0.75, is a team
    that moved the ball 8.75 points *worse* than it scored."""
    epa = {"g1": GameEpa(home=0.15, away=0.05, home_plays=70, away_plays=70)}
    played = game("Home", "Away", 31, 10, game_id="g1")

    predictor = _predictor(
        epa=epa, play_weight=1.0, epa_share=1.0, epa_residual_beta=0.75
    )

    assert predictor._epa_share_of_game(played) == pytest.approx(
        1 / (1 + math.exp(-(7.0 - 0.75 * 21) / 10.0))
    )


def test_two_teams_that_won_by_the_same_split_on_the_residual(
    game: GameFactory,
) -> None:
    """The sentence the residual exists to say.

    Same scoreline, same EPA per play -- but one offense was on the field
    long enough to earn it and the other was not. Without the residual the
    scoreboard drowns that; with it, the two games are different evidence.
    """
    epa = {
        "earned": GameEpa(home=0.30, away=0.00, home_plays=75, away_plays=60),
        "lucky": GameEpa(home=0.02, away=0.10, home_plays=55, away_plays=75),
    }
    predictor = _predictor(
        epa=epa, play_weight=1.0, epa_share=1.0, epa_residual_beta=0.75
    )

    earned = predictor._epa_share_of_game(game("H", "A", 24, 14, game_id="earned"))
    lucky = predictor._epa_share_of_game(game("H", "A", 24, 14, game_id="lucky"))

    assert earned is not None and lucky is not None
    assert earned > lucky


def test_a_negative_residual_beta_adds_the_scoreboard_back() -> None:
    """Which is the one thing this parameter must not be allowed to do."""
    with pytest.raises(ValueError, match="epa_residual_beta"):
        _predictor(epa_residual_beta=-0.1)


def test_the_residual_is_not_reachable_by_the_weights(game: GameFactory) -> None:
    """Why this is a parameter rather than something the blend already spans.

    The blend is convex, so it can average its sources and never difference
    them. A residual model and a no-residual model are therefore different
    functions of the same two inputs -- if they weren't, this parameter would
    be redundant with `epa_share`.
    """
    epa = {"g1": GameEpa(home=0.30, away=0.00, home_plays=70, away_plays=70)}
    played = game("Home", "Away", 35, 7, game_id="g1")

    with_residual = _predictor(
        epa=epa, play_weight=0.5, epa_share=1.0, epa_residual_beta=0.75
    )._actual(played)
    without = _predictor(epa=epa, play_weight=0.5, epa_share=1.0)._actual(played)

    assert with_residual != pytest.approx(without)
    # And the residual reads a dominant scoreline as the underperformance it
    # was, so it pulls the target *below* the plain blend.
    assert with_residual < without


def test_a_release_from_before_mov_scale_still_loads() -> None:
    """`scoring_method` was in this class's state until `mov_scale` replaced
    the parent's scorer, and releases carrying it have to stay readable.

    Safe at the defaults and only there: every such release was published on
    sigmoid, which is `mov_scale` at 10, which is the default a missing one
    gets.
    """
    legacy = {
        "league": "test_league",
        "home_advantage": 95,
        "k": 65,
        "weekly_rd_increase": 1,
        "season_rd_increase": 120,
        "initial_rd": 216,
        "scoring_method": "sigmoid",
        "season_regression": 0.0,
        "play_weight": 0.17,
        "epa_share": 0.29,
        "epa_margin_scale": 10.0,
        "ratings": {},
        "anchors": {},
    }

    restored = BlendedGlickoPredictor.from_state_dict(legacy)

    state = restored.state_dict()
    assert state["mov_scale"] == DEFAULT_MOV_SCALE
    assert state["control_temp"] == DEFAULT_CONTROL_TEMP
    assert state["epa_residual_beta"] == DEFAULT_EPA_RESIDUAL_BETA
    assert state["play_weight"] == pytest.approx(0.17)
    assert "scoring_method" not in state
