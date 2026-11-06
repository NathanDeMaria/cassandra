from datetime import date
from pathlib import Path

import pytest

from cassandra.players import read_players_file

_HEADER = (
    "league,season,as_of,athlete_id,player,team_id,team,projected,offense,defense,"
    "total,offense_sd,defense_sd,total_sd,vs_team,games,starts,minutes,share\n"
)


def test_a_mornings_file_reads_played_and_waiting_alike(tmp_path: Path) -> None:
    path = tmp_path / "mens_players_2026.csv"
    path.write_text(
        _HEADER
        + "mens,2026,2026-11-05,1,Pat,9,Team,false,1,2,3,1,1,1.4,2.5,4,4,100,0.5\n"
        + "mens,2026,2026-11-05,2,,9,,true,0,1,1,2,2,2.8,0.5,0,0,0,0.25\n"
    )
    stored = read_players_file(path)
    assert (stored.league, stored.season) == ("mens", 2026)
    assert stored.as_of == date(2026, 11, 5)
    played, waiting = stored.players
    assert played.player == "Pat" and not played.projected and played.games == 4
    assert waiting.player is None and waiting.team is None and waiting.projected


def test_a_finished_season_has_no_morning_and_may_miss_a_box_score(
    tmp_path: Path,
) -> None:
    path = tmp_path / "womens_players_2019.csv"
    path.write_text(_HEADER + "womens,2019,,1,Pat,9,Team,false,1,2,3,1,1,1.4,2.5,,,,\n")
    stored = read_players_file(path)
    assert stored.as_of is None
    assert stored.players[0].share is None and stored.players[0].games is None


def test_a_file_named_for_something_else_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "mens_2026_roster.csv"
    path.write_text(_HEADER)
    with pytest.raises(ValueError):
        read_players_file(path)
