"""Tests for the quarterback availability index."""

from datetime import datetime

import pytest

from .qb_out import (
    QB_OUT_DECAY,
    QbOutFile,
    QbOutIndex,
    run_weights,
    validated_qb_out_penalty,
)


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


def test_a_team_can_be_assumed_out_for_a_block() -> None:
    """What `picks.py` asks after the replay: the price if his starter sits."""
    index = QbOutIndex({"g1": ["B"]})

    with index.assuming("g1", "A"):
        assert index.differential(_Matchup("A", "B")) == 0.0
    with index.assuming("g2", "A"):
        assert index.is_out("g2", "A")

    assert index.differential(_Matchup("A", "B")) == 1.0
    assert not index.is_out("g2", "A")
    assert len(index) == 1


def test_a_long_absence_weighs_less_each_game() -> None:
    weights = run_weights({"g1": {"B": 1}, "g2": {"B": 3}})
    index = QbOutIndex({"g1": ["B"], "g2": ["B"]}, weights)

    assert index.differential(_Matchup("A", "B", "g1")) == 1.0
    assert index.differential(_Matchup("A", "B", "g2")) == pytest.approx(
        QB_OUT_DECAY**2
    )


def test_a_fixture_can_carry_a_chance_without_being_out() -> None:
    """What `picks.py` lays on: a starter who may sit, priced, not flagged."""
    index = QbOutIndex({"g1": ["B"]}).with_weights({"fix": {"A": 0.4}})

    assert not index.is_out("fix", "A")
    assert index.differential(_Matchup("A", "B", "fix")) == pytest.approx(-0.4)
    assert index.differential(_Matchup("A", "B", "g1")) == 1.0


def test_assuming_him_out_counts_him_whole() -> None:
    index = QbOutIndex(weights={"fix": {"A": 0.4}})

    with index.assuming("fix", "A"):
        assert index.differential(_Matchup("A", "B", "fix")) == -1.0
    assert index.differential(_Matchup("A", "B", "fix")) == pytest.approx(-0.4)
    assert not index.is_out("fix", "A")


def test_a_release_file_reads_as_the_whole_index() -> None:
    """What a consumer with no `~/.cassandra` builds from `qb_out.json`."""
    stored = QbOutFile(
        league="ncaafb",
        games={"g1": ["B"], "g2": ["B"]},
        runs={"g2": {"B": 2}},
        fixtures={"next": {"A": 0.5}},
    )
    index = QbOutIndex.from_file(stored)

    assert index.differential(_Matchup("A", "B", "g1")) == 1.0
    assert index.differential(_Matchup("A", "B", "g2")) == pytest.approx(QB_OUT_DECAY)
    assert index.differential(_Matchup("A", "B", "next")) == pytest.approx(-0.5)
    assert not index.is_out("next", "A")


def test_an_older_file_has_no_runs_or_fixtures() -> None:
    stored = QbOutFile.model_validate_json('{"league": "nfl", "games": {"g1": ["B"]}}')

    assert QbOutIndex.from_file(stored).differential(_Matchup("A", "B")) == 1.0


def test_a_league_with_no_saved_index_gets_an_empty_one() -> None:
    """Four of the six leagues aren't football."""
    assert len(QbOutIndex.for_league("no-such-league")) == 0


@pytest.mark.parametrize("value", [-0.1, -50.0])
def test_a_negative_penalty_is_refused(value: float) -> None:
    """It would mean a team improves by losing its quarterback."""
    with pytest.raises(ValueError, match="qb_out_penalty must be non-negative"):
        validated_qb_out_penalty(value)
