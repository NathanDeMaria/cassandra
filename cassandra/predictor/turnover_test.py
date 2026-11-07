from pathlib import Path

import pytest

from . import turnover
from .turnover import TurnoverIndex, load_turnover


def test_a_team_is_measured_against_its_seasons_average() -> None:
    index = TurnoverIndex({2024: {"A": 0.8, "B": 0.4}, 2025: {"A": 0.5}})
    assert index.above_average("A", 2024) == pytest.approx(0.2)
    assert index.above_average("B", 2024) == pytest.approx(-0.2)
    # alone in its season, it's the average
    assert index.above_average("A", 2025) == 0


def test_a_team_or_season_not_on_file_has_no_answer() -> None:
    index = TurnoverIndex({2024: {"A": 0.8}})
    assert index.above_average("B", 2024) is None
    assert index.above_average("A", 2023) is None
    assert TurnoverIndex().above_average("A", 2024) is None


def test_the_file_reads_returning_minutes_as_turnover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "nfl_turnover.csv"
    path.write_text(
        "league,season,team_id,team,minutes,returning\n"
        "nfl,2025,1,Home U,1000,250\n"
        "nfl,2025,2,Away St,0,0\n"
    )
    monkeypatch.setattr(turnover, "turnover_path", lambda league: path)
    load_turnover.cache_clear()
    try:
        assert load_turnover("nfl") == {2025: {"Home U": 0.75}}
    finally:
        load_turnover.cache_clear()


def test_a_league_with_no_file_has_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(turnover, "turnover_path", lambda league: tmp_path / "no.csv")
    load_turnover.cache_clear()
    try:
        assert load_turnover("nfl") == {}
    finally:
        load_turnover.cache_clear()
