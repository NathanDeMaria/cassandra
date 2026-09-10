"""Tests for the quarterback availability index."""

from datetime import datetime

import pytest

from .qb_out import QbOutIndex, validated_qb_out_penalty


class _Matchup:
    def __init__(self, home: str, away: str, game_id: str = "g1") -> None:
        self.home, self.away = home, away
        self.neutral_site = False
        self.game_id = game_id
        self.date = datetime(2026, 9, 12)


def test_an_empty_index_has_nobody_out() -> None:
    index = QbOutIndex()

    assert not index.is_out("g1", "A")
    assert index.differential(_Matchup("A", "B")) == 0.0


def test_a_game_nobody_has_played_has_nobody_out() -> None:
    """The live-prediction default: assume both quarterbacks are fine."""
    index = QbOutIndex({"g1": ["A"]})

    assert not index.is_out("fixture-42", "A")


def test_the_differential_is_signed_toward_the_home_team() -> None:
    assert QbOutIndex({"g1": ["B"]}).differential(_Matchup("A", "B")) == 1.0
    assert QbOutIndex({"g1": ["A"]}).differential(_Matchup("A", "B")) == -1.0
    assert QbOutIndex({"g1": ["A", "B"]}).differential(_Matchup("A", "B")) == 0.0


def test_an_index_can_be_built_in_memory_for_a_fixture() -> None:
    """What a caller holding an injury report passes in."""
    index = QbOutIndex({"401752708": ["LSU Tigers"]})

    assert index.is_out("401752708", "LSU Tigers")
    assert not index.is_out("401752708", "Florida Gators")
    assert len(index) == 1


def test_a_league_with_no_saved_index_gets_an_empty_one() -> None:
    """Four of the six leagues aren't football."""
    assert len(QbOutIndex.for_league("no-such-league")) == 0


@pytest.mark.parametrize("value", [-0.1, -50.0])
def test_a_negative_penalty_is_refused(value: float) -> None:
    """It would mean a team improves by losing its quarterback."""
    with pytest.raises(ValueError, match="qb_out_penalty must be non-negative"):
        validated_qb_out_penalty(value)
