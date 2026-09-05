"""The EPA index: the artifact, and the margin it converts to.

Nothing here reads a play. What the index does is hold four numbers per game
and turn them into a margin, and the arithmetic in that conversion is the
part with a `1 -`-shaped mistake available in it.
"""

import json
from pathlib import Path

import pytest

from .epa import EpaFile, EpaFit, EpaIndex, load_epa, read_epa_file
from .types import GameEpa

_FIT = EpaFit(
    lucky_ones="abc123",
    run_id="20260904-023610",
    ep_run_id="20260904-024017",
    clip=3.0,
    reading="unweighted",
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the artifact at tmp_path, and clear the cached read around it."""
    monkeypatch.setattr(
        "cassandra.predictor.epa.epa_path",
        lambda league: tmp_path / f"{league}_epa.json",
    )
    load_epa.cache_clear()
    yield
    load_epa.cache_clear()


def _write(tmp_path: Path, league: str, games: dict[str, GameEpa]) -> None:
    document = EpaFile(league=league, fit=_FIT, games=games)
    (tmp_path / f"{league}_epa.json").write_text(
        json.dumps(document.model_dump(mode="json"))
    )


def test_a_league_with_no_sweep_reads_as_empty() -> None:
    """Four of the six leagues aren't football. That's ordinary, not an error."""
    assert load_epa("nhl") == {}
    assert read_epa_file("nhl") is None
    assert len(EpaIndex.for_league("nhl")) == 0


def test_a_written_index_is_what_a_predictor_reads_back(tmp_path: Path) -> None:
    _write(tmp_path, "nfl", {"g1": GameEpa(0.12, -0.04, 68, 61)})

    index = EpaIndex.for_league("nfl")

    assert len(index) == 1
    assert index.get("g1") == GameEpa(0.12, -0.04, 68, 61)


def test_the_index_copies_rather_than_holding_the_cached_mapping(
    tmp_path: Path,
) -> None:
    """`load_epa` is cached, so every probe in a search gets one mapping.

    An index that held it would leak a top-up into every other predictor
    built in the same process.
    """
    _write(tmp_path, "nfl", {"g1": GameEpa(0.1, 0.0, 60, 60)})
    first = EpaIndex.for_league("nfl")

    first._epa["g2"] = GameEpa(0.5, 0.0, 60, 60)

    assert EpaIndex.for_league("nfl").get("g2") is None


def test_a_game_with_nothing_stored_has_no_margin() -> None:
    """Most of an NCAAFB schedule, and all of football before 2006."""
    assert EpaIndex().margin("g1") is None


def test_the_margin_is_the_points_each_offense_added() -> None:
    """Per-play averages back out through their own play counts.

    0.10 over 80 snaps is 8 points added; -0.05 over 60 is 3 given up. The
    home team out-produced the away team by 11.
    """
    index = EpaIndex({"g1": GameEpa(home=0.10, away=-0.05, home_plays=80, away_plays=60)})

    assert index.margin("g1") == pytest.approx(11.0)


def test_the_play_counts_matter_and_not_just_the_averages() -> None:
    """The whole reason the counts are in the artifact.

    Two games where the home team averaged more per play than the away team
    by the same amount, and they are not worth the same margin -- the one
    where the home offense was on the field twice as long produced more
    points. Differencing the averages alone would call these identical.
    """
    short = EpaIndex({"g": GameEpa(0.2, 0.1, 30, 30)}).margin("g")
    long = EpaIndex({"g": GameEpa(0.2, 0.1, 60, 60)}).margin("g")
    assert short is not None and long is not None

    assert long == pytest.approx(2 * short)


def test_a_game_nobody_moved_the_ball_in_is_a_zero_margin() -> None:
    index = EpaIndex({"g1": GameEpa(0.0, 0.0, 70, 70)})

    assert index.margin("g1") == pytest.approx(0.0)


def test_the_sides_are_not_a_share_and_can_both_be_positive() -> None:
    """Unlike `GameControl`, which is one number and its complement.

    A shootout is two offenses that both added points, and the margin between
    them is small. Nothing here should read the away number as `1 - home`.
    """
    index = EpaIndex({"g1": GameEpa(0.30, 0.28, 70, 70)})

    assert index.margin("g1") == pytest.approx(0.02 * 70)


def test_the_header_says_which_models_and_which_reading(tmp_path: Path) -> None:
    """What makes a stale index detectable rather than merged into."""
    _write(tmp_path, "nfl", {"g1": GameEpa(0.1, 0.0, 60, 60)})

    stored = read_epa_file("nfl")

    assert stored is not None
    assert stored.fit == _FIT
    assert stored.fit.run_id != stored.fit.ep_run_id


def test_the_fit_requires_every_field() -> None:
    """No defaults, because there are no older files to stay honest about.

    A `reading` that defaulted would make an index built under a different
    one claim to be this one, which is the failure `ControlFit.reading`'s
    default is carefully arranged to avoid.
    """
    with pytest.raises(ValueError):
        EpaFit(lucky_ones="abc", run_id="1", ep_run_id="2", clip=3.0)  # ty: ignore[missing-argument]
