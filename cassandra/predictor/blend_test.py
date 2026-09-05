"""The weighting both play-by-play models share.

Scale-free on purpose: `combine` takes three numbers already on one scale and
never asks what scale that is, so these read as plain arithmetic. What each
model converts before calling it is checked in that model's own test file.
"""

import pytest

from .blend import PlayBlend, validated_fraction, validated_scale


def test_the_three_weights_always_sum_to_one() -> None:
    blend = PlayBlend(play_weight=0.6, epa_share=0.25)

    assert blend.control_weight == pytest.approx(0.45)
    assert blend.epa_weight == pytest.approx(0.15)
    scoreboard = 1 - blend.play_weight
    assert scoreboard + blend.control_weight + blend.epa_weight == pytest.approx(1.0)


@pytest.mark.parametrize("play_weight", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("epa_share", [0.0, 0.5, 1.0])
def test_every_corner_of_the_box_is_a_valid_blend(
    play_weight: float, epa_share: float
) -> None:
    """The reason the search gets two knobs instead of three weights.

    A convex combination of three numbers in [0, 1] is in [0, 1], everywhere
    in the box the optimizer can propose.
    """
    blend = PlayBlend(play_weight=play_weight, epa_share=epa_share)

    assert 0.0 <= blend.combine(1.0, 0.0, 0.0) <= 1.0
    assert 0.0 <= blend.combine(0.0, 1.0, 1.0) <= 1.0


def test_no_play_weight_returns_the_scoreboard_untouched() -> None:
    """Exactly, so a search that finds the plays worthless recovers the plain
    model rather than something within rounding of it."""
    blend = PlayBlend(play_weight=0.0, epa_share=0.5)

    assert blend.combine(0.375, 1.0, 1.0) == 0.375
    assert blend.uses_plays is False


def test_a_full_play_weight_drops_the_scoreboard() -> None:
    blend = PlayBlend(play_weight=1.0, epa_share=0.5)

    assert blend.combine(1.0, 0.4, 0.6) == pytest.approx(0.5)


def test_the_combination_is_the_weights_written_out() -> None:
    blend = PlayBlend(play_weight=0.5, epa_share=0.5)

    assert blend.combine(1.0, 0.5, 0.0) == pytest.approx(
        0.5 * 1.0 + 0.25 * 0.5 + 0.25 * 0.0
    )


def test_a_missing_signal_renormalizes_over_the_rest() -> None:
    """Not a vote for zero, which is what dropping the term would make it.

    With EPA gone the scoreboard and control keep their 2:1 ratio and split
    the whole weight, rather than the missing quarter pulling the target
    toward 0.
    """
    blend = PlayBlend(play_weight=0.5, epa_share=0.5)

    assert blend.combine(1.0, 0.5, None) == pytest.approx((0.5 * 1.0 + 0.25 * 0.5) / 0.75)


def test_a_game_with_neither_signal_is_its_scoreboard() -> None:
    """Most of an NCAAFB schedule, and all of football before 2006."""
    for play_weight in (0.25, 0.5, 1.0):
        blend = PlayBlend(play_weight=play_weight, epa_share=0.5)
        assert blend.combine(0.8, None, None) == pytest.approx(0.8)


def test_a_full_play_weight_with_one_signal_uses_only_that_one() -> None:
    """The case that would divide by zero if the renormalizer were naive."""
    blend = PlayBlend(play_weight=1.0, epa_share=0.5)

    assert blend.combine(1.0, 0.3, None) == pytest.approx(0.3)
    assert blend.combine(1.0, None, 0.3) == pytest.approx(0.3)


def test_an_epa_share_of_one_ignores_control() -> None:
    blend = PlayBlend(play_weight=1.0, epa_share=1.0)

    assert blend.combine(1.0, 0.0, 0.7) == pytest.approx(0.7)


def test_an_epa_share_of_zero_ignores_epa() -> None:
    blend = PlayBlend(play_weight=1.0, epa_share=0.0)

    assert blend.combine(1.0, 0.7, 0.0) == pytest.approx(0.7)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_a_knob_outside_the_unit_interval_is_not_a_blend(value: float) -> None:
    with pytest.raises(ValueError, match="knob"):
        validated_fraction("knob", value)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_scale_has_to_be_positive(value: float) -> None:
    """0 is a second spelling of a weight of 0, and two spellings of one model
    make a fitted config ambiguous about which the search found."""
    with pytest.raises(ValueError, match="scale"):
        validated_scale("scale", value)


def test_validated_passes_the_pair_through() -> None:
    assert PlayBlend.validated(0.3, 0.7) == PlayBlend(0.3, 0.7)
