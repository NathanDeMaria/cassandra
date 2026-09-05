"""What the blend does to the margin a rating chases.

Every test here builds its indexes by hand rather than reading a sweep: what
is being checked is the arithmetic that combines three margins and the rule
for a game that only has some of them, and a real index would make those
assertions depend on a win probability fit.
"""

import pytest

from .conftest import GameFactory
from .epa import EpaIndex
from .game_control import GameControlIndex
from .margin_blend import (
    DEFAULT_CONTROL_SCALE,
    DEFAULT_EPA_SCALE,
    DEFAULT_EPA_SHARE,
    DEFAULT_PLAY_WEIGHT,
    BlendedMarginEloPredictor,
)
from .margin_elo import MarginEloPredictor
from .types import GameControl, GameEpa

_CONTROL_SCALE = DEFAULT_CONTROL_SCALE


def _predictor(
    control: dict[str, GameControl] | None = None,
    epa: dict[str, GameEpa] | None = None,
    *,
    play_weight: float = DEFAULT_PLAY_WEIGHT,
    epa_share: float = DEFAULT_EPA_SHARE,
    control_scale: float = DEFAULT_CONTROL_SCALE,
    epa_scale: float = DEFAULT_EPA_SCALE,
    mov_cap: float = 40.0,
) -> BlendedMarginEloPredictor:
    """The model over indexes built by hand, with the knobs spelled out.

    Every parameter named rather than forwarded as `**params`: the two index
    arguments take mappings and the rest take floats, and a `**kwargs` helper
    is one typo away from passing a weight as a set of ratings.
    """
    return BlendedMarginEloPredictor(
        "test_league",
        game_control=GameControlIndex(control or {}),
        game_epa=EpaIndex(epa or {}),
        play_weight=play_weight,
        epa_share=epa_share,
        control_scale=control_scale,
        epa_scale=epa_scale,
        mov_cap=mov_cap,
    )


def test_the_weights_are_a_simplex_the_search_cannot_leave() -> None:
    """Two knobs in the unit box, three weights that always sum to 1."""
    predictor = _predictor(play_weight=0.6, epa_share=0.25)

    assert predictor.control_weight == pytest.approx(0.45)
    assert predictor.epa_weight == pytest.approx(0.15)
    assert predictor.control_weight + predictor.epa_weight == pytest.approx(0.6)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_a_play_weight_outside_the_unit_interval_is_not_a_blend(
    value: float,
) -> None:
    """Below 0 learns backwards; above 1 overshoots the signal."""
    with pytest.raises(ValueError, match="play_weight"):
        _predictor(play_weight=value)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_an_epa_share_outside_the_unit_interval_is_not_a_split(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="epa_share"):
        _predictor(epa_share=value)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_the_control_scale_has_to_be_positive(value: float) -> None:
    """0 is a second spelling of a weight of 0, and two spellings of one
    model make a fitted config ambiguous. Negative reads control backwards."""
    with pytest.raises(ValueError, match="control_scale"):
        _predictor(control_scale=value)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_the_epa_scale_has_to_be_positive(value: float) -> None:
    with pytest.raises(ValueError, match="epa_scale"):
        _predictor(epa_scale=value)


def test_no_play_weight_is_the_parent_exactly(game: GameFactory) -> None:
    """A search that finds the plays worthless recovers `MarginEloPredictor`.

    Exactly, not approximately -- and with an index full of numbers loud
    enough to show up if the short-circuit ever stopped short-circuiting.
    """
    played = game("Home", "Away", 24, 10, game_id="g1")
    control = {"g1": GameControl(home=0.05, seconds=3600)}
    epa = {"g1": GameEpa(home=-0.5, away=0.5, home_plays=70, away_plays=70)}

    blended = _predictor(control, epa, play_weight=0.0)
    plain = MarginEloPredictor("test_league")
    blended.update_game(played)
    plain.update_game(played)

    assert blended.get_rating("Home") == plain.get_rating("Home")


def test_a_game_with_no_play_by_play_keeps_its_own_margin(
    game: GameFactory,
) -> None:
    """Most of an NCAAFB schedule, and all of it before 2006.

    The weights renormalize over what a game actually has, so a game with
    nothing needs no fallback rule -- it is rated on its final margin at
    full weight, and at `play_weight=1` too, which is the setting that would
    otherwise leave it with no target at all.
    """
    played = game("Home", "Away", 24, 10, game_id="unswept")

    for weight in (0.5, 1.0):
        blended = _predictor(play_weight=weight)
        plain = MarginEloPredictor("test_league")
        blended.update_game(played)
        plain.update_game(played)
        assert blended.get_rating("Home") == plain.get_rating("Home")


def test_control_converts_linearly_and_symmetrically() -> None:
    """A game controlled 0.5 is a tied game; 1.0 is the whole scale."""
    predictor = _predictor(
        {
            "even": GameControl(home=0.5, seconds=3600),
            "all": GameControl(home=1.0, seconds=3600),
            "none": GameControl(home=0.0, seconds=3600),
        },
        control_scale=_CONTROL_SCALE,
    )

    assert predictor._control_margin("even") == pytest.approx(0.0)
    assert predictor._control_margin("all") == pytest.approx(_CONTROL_SCALE)
    assert predictor._control_margin("none") == pytest.approx(-_CONTROL_SCALE)


def test_epa_is_taken_at_face_value_by_default() -> None:
    """`epa_scale` of 1 means the points EPA claims are the points it gets."""
    epa = {"g1": GameEpa(home=0.1, away=0.0, home_plays=70, away_plays=70)}

    assert _predictor(epa=epa)._epa_margin("g1") == pytest.approx(7.0)
    assert _predictor(epa=epa, epa_scale=2.0)._epa_margin("g1") == pytest.approx(14.0)


def test_the_target_is_the_convex_combination_of_the_three(
    game: GameFactory,
) -> None:
    """The whole model, checked against the arithmetic written out.

    Half the final margin, a quarter each of what the two signals imply.
    """
    control = {"g1": GameControl(home=0.75, seconds=3600)}
    epa = {"g1": GameEpa(home=0.2, away=0.0, home_plays=70, away_plays=70)}
    predictor = _predictor(
        control, epa, play_weight=0.5, epa_share=0.5, control_scale=_CONTROL_SCALE
    )
    played = game("Home", "Away", 24, 10, game_id="g1")

    # 0.5 * 14 + 0.25 * ((2 * 0.75 - 1) * 30) + 0.25 * (0.2 * 70)
    assert predictor._actual_margin(played) == pytest.approx(
        0.5 * 14 + 0.25 * 15.0 + 0.25 * 14.0
    )


def test_a_game_missing_one_signal_renormalizes_over_the_rest(
    game: GameFactory,
) -> None:
    """Half of a football schedule has control but no EPA or the reverse only
    if a sweep half-ran; the general rule is what's checked.

    With EPA missing, the scoreboard and control split the whole target in
    the ratio their weights had, rather than the EPA share silently becoming
    a vote for zero.
    """
    control = {"g1": GameControl(home=0.75, seconds=3600)}
    predictor = _predictor(
        control, play_weight=0.5, epa_share=0.5, control_scale=_CONTROL_SCALE
    )
    played = game("Home", "Away", 24, 10, game_id="g1")

    # Weights 0.5 and 0.25 renormalize to 2/3 and 1/3.
    assert predictor._actual_margin(played) == pytest.approx(
        (0.5 * 14 + 0.25 * 15.0) / 0.75
    )


def test_an_epa_share_of_zero_ignores_epa_entirely(game: GameFactory) -> None:
    """The blend reduces to margin-and-control, which is the model
    `ControlGlickoPredictor` is on the other scale."""
    control = {"g1": GameControl(home=0.75, seconds=3600)}
    epa = {"g1": GameEpa(home=9.0, away=0.0, home_plays=70, away_plays=70)}
    played = game("Home", "Away", 24, 10, game_id="g1")

    with_epa = _predictor(control, epa, play_weight=0.5, epa_share=0.0)
    without = _predictor(control, play_weight=0.5, epa_share=0.0)

    assert with_epa._actual_margin(played) == without._actual_margin(played)


def test_the_blend_is_capped_on_the_way_out(game: GameFactory) -> None:
    """A play-derived margin past `mov_cap` doesn't get to be past it.

    The scoreboard's own margin is capped by the parent before the blend, so
    this is the case the second cap exists for: an enormous EPA margin at
    full play weight.
    """
    epa = {"g1": GameEpa(home=2.0, away=0.0, home_plays=70, away_plays=70)}
    predictor = _predictor(
        epa=epa, play_weight=1.0, epa_share=1.0, mov_cap=40.0
    )
    played = game("Home", "Away", 24, 10, game_id="g1")

    assert predictor._actual_margin(played) == pytest.approx(40.0)


def test_a_controlled_game_rates_higher_than_a_stolen_one(
    game: GameFactory,
) -> None:
    """The point of the model, in one comparison.

    Two teams win by the same three points. One controlled the game and
    out-produced its opponent; the other was behind all afternoon and won it
    late. The blend rates the first higher, and `MarginEloPredictor` -- which
    sees only the score -- rates them the same.
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
        ratings.append(predictor.get_rating("Home"))

    assert ratings[0] > ratings[1]

    plain = [
        MarginEloPredictor("test_league") for _ in range(2)
    ]
    for predictor, game_id in zip(plain, ("earned", "stolen")):
        predictor.update_game(game("Home", "Away", 17, 14, game_id=game_id))
    assert plain[0].get_rating("Home") == plain[1].get_rating("Home")


def test_the_indexes_stay_out_of_the_state_dict() -> None:
    """Tens of thousands of floats against a release that is a page of
    ratings."""
    predictor = _predictor(
        {"g1": GameControl(home=0.6, seconds=3600)},
        {"g1": GameEpa(0.1, 0.0, 70, 70)},
        play_weight=0.4,
        epa_share=0.3,
    )

    state = predictor.state_dict()

    assert state["play_weight"] == pytest.approx(0.4)
    assert state["epa_share"] == pytest.approx(0.3)
    assert "game_control" not in state
    assert "game_epa" not in state
