from pathlib import Path

import pytest

from . import roster
from .roster import ACADEMIES, RETURNING, TALENT, RosterIndex, SeasonValue, standardized


def _row(season: int, espn_id: str, value: float) -> SeasonValue:
    return SeasonValue(season, espn_id, f"team {espn_id}", value)


def test_talent_is_standardized_within_its_seasons_division() -> None:
    rows = [
        _row(2020, "1", 900.0),
        _row(2020, "2", 500.0),
        _row(2020, "3", 100.0),  # a lower-division team: not read, not averaged
        # A season on a narrower scale: the same standing, half the points.
        _row(2021, "1", 800.0),
        _row(2021, "2", 600.0),
    ]
    got = standardized(rows, lambda espn_id, season: espn_id != "3")
    assert got == {
        ("1", 2020): pytest.approx(1.0),
        ("2", 2020): pytest.approx(-1.0),
        ("1", 2021): pytest.approx(1.0),
        ("2", 2021): pytest.approx(-1.0),
    }


def test_the_academies_are_left_out() -> None:
    academy = next(iter(ACADEMIES))
    rows = [_row(2020, "1", 600.0), _row(2020, "2", 400.0), _row(2020, academy, 70.0)]
    got = standardized(rows, lambda espn_id, season: True, TALENT.excluded)
    assert got == {("1", 2020): pytest.approx(1.0), ("2", 2020): pytest.approx(-1.0)}
    # Returning production reads them like anyone.
    assert (academy, 2020) in standardized(
        rows, lambda espn_id, season: True, RETURNING.excluded
    )


def test_a_league_without_a_file_has_no_talent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        roster, "roster_path", lambda league, measure: tmp_path / "none.csv"
    )
    roster.load_roster.cache_clear()
    try:
        assert roster.load_roster("ncaafb", TALENT) == {}
        assert len(RosterIndex.for_league("ncaafb", TALENT)) == 0
    finally:
        roster.load_roster.cache_clear()


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
    monkeypatch.setattr(roster, "roster_path", lambda league, measure: path)
    roster.load_roster.cache_clear()
    try:
        index = RosterIndex.for_league("ncaafb", TALENT)
    finally:
        roster.load_roster.cache_clear()
    assert index.get("Alabama Crimson Tide", 2020) == pytest.approx(1.0)
    assert index.get("Troy Trojans", 2020) == pytest.approx(-1.0)
    assert index.get("Air Force Falcons", 2020) is None
    assert dict(index.season(2020)) == {
        "Alabama Crimson Tide": pytest.approx(1.0),
        "Troy Trojans": pytest.approx(-1.0),
    }
    assert list(index.season(None)) == []


def test_previous_is_the_latest_earlier_rated_season() -> None:
    index = RosterIndex({("A", 2015): 1.0, ("A", 2017): 2.0, ("B", 2017): -1.0})
    assert index.previous("A", 2018) == 2.0
    assert index.previous("A", 2017) == 1.0  # 2016 unrated: the season before it
    assert index.previous("A", 2015) == 0.0
    assert index.previous("C", 2017) == 0.0


def test_returning_production_reads_the_usage_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "ncaafb_returning.csv"
    path.write_text(
        "season,espn_id,team,percent_ppa,usage\n"
        "2020,333,Alabama Crimson Tide,9.40,0.80\n"
        "2020,2653,Troy Trojans,0.50,0.40\n"
        "2020,2005,Air Force Falcons,0.60,0.60\n"
    )
    monkeypatch.setattr(roster, "roster_path", lambda league, measure: path)
    roster.load_roster.cache_clear()
    try:
        index = RosterIndex.for_league("ncaafb", RETURNING)
    finally:
        roster.load_roster.cache_clear()
    sd = (((0.8 - 0.6) ** 2 + (0.4 - 0.6) ** 2 + 0) / 3) ** 0.5
    assert index.get("Alabama Crimson Tide", 2020) == pytest.approx(0.2 / sd)
    assert index.get("Air Force Falcons", 2020) == pytest.approx(0.0)
