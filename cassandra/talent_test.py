from pathlib import Path

import pytest

from . import talent
from .talent import ACADEMIES, TalentIndex, TalentRow, above_average


def _row(season: int, espn_id: str, value: float) -> TalentRow:
    return TalentRow(season, espn_id, f"team {espn_id}", value)


def test_talent_is_read_against_its_seasons_division_average() -> None:
    rows = [
        _row(2020, "1", 900.0),
        _row(2020, "2", 500.0),
        _row(2020, "3", 100.0),  # a lower-division team: not read, not averaged
        _row(2021, "1", 800.0),
        _row(2021, "2", 800.0),
    ]
    got = above_average(rows, lambda espn_id, season: espn_id != "3")
    assert got == {
        ("1", 2020): pytest.approx(2.0),
        ("2", 2020): pytest.approx(-2.0),
        ("1", 2021): pytest.approx(0.0),
        ("2", 2021): pytest.approx(0.0),
    }


def test_the_academies_are_left_out() -> None:
    academy = next(iter(ACADEMIES))
    rows = [_row(2020, "1", 600.0), _row(2020, "2", 400.0), _row(2020, academy, 70.0)]
    got = above_average(rows, lambda espn_id, season: True)
    assert got == {("1", 2020): pytest.approx(1.0), ("2", 2020): pytest.approx(-1.0)}


def test_a_league_without_a_file_has_no_talent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(talent, "talent_path", lambda league: tmp_path / "none.csv")
    talent.load_talent.cache_clear()
    try:
        assert talent.load_talent("ncaafb") == {}
        assert len(TalentIndex.for_league("ncaafb")) == 0
    finally:
        talent.load_talent.cache_clear()


def test_the_file_is_keyed_by_the_name_a_matchup_carries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "ncaafb_talent.csv"
    path.write_text(
        "season,espn_id,team,talent\n"
        "2020,333,Alabama Crimson Tide,980.00\n"
        "2020,2653,Troy Trojans,480.00\n"
        "2020,2005,Air Force Falcons,40.00\n"
    )
    monkeypatch.setattr(talent, "talent_path", lambda league: path)
    talent.load_talent.cache_clear()
    try:
        index = TalentIndex.for_league("ncaafb")
    finally:
        talent.load_talent.cache_clear()
    assert index.get("Alabama Crimson Tide", 2020) == pytest.approx(2.5)
    assert index.get("Troy Trojans", 2020) == pytest.approx(-2.5)
    assert index.get("Air Force Falcons", 2020) is None
    assert dict(index.season(2020)) == {
        "Alabama Crimson Tide": pytest.approx(2.5),
        "Troy Trojans": pytest.approx(-2.5),
    }
    assert list(index.season(None)) == []
