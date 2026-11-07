import asyncio
from pathlib import Path

import pytest

from cassandra import turnover_build
from cassandra.players import PlayerRow, PlayersFile
from cassandra.predictor.turnover import load_turnover
from cassandra.turnover_build import TeamReturning, every_season, rebuild, returning


def _player(
    athlete: str,
    team_id: str = "1",
    minutes: float | None = 100,
    projected: bool = False,
    team: str | None = "Home U",
) -> PlayerRow:
    return PlayerRow(
        athlete_id=athlete,
        team_id=team_id,
        team=team,
        projected=projected,
        offense=0,
        defense=0,
        total=0,
        offense_sd=1,
        defense_sd=1,
        total_sd=1,
        vs_team=0,
        minutes=minutes,
    )


def _table(season: int, *players: PlayerRow) -> PlayersFile:
    return PlayersFile(league="mens", season=season, players=list(players))


def _row(minutes: float, back: float, season: int = 2025) -> TeamReturning:
    return TeamReturning("mens", season, "1", "Home U", minutes, back)


def test_only_the_minutes_of_players_still_on_the_team_return() -> None:
    before = _table(2024, _player("stays", minutes=300), _player("gone", minutes=700))
    assert returning(before, _table(2025, _player("stays"), _player("new"))) == [
        _row(1000, 300)
    ]


def test_a_rostered_player_who_hasnt_played_has_returned() -> None:
    before = _table(2024, _player("hurt", minutes=300), _player("gone", minutes=700))
    now = _table(2025, _player("hurt", minutes=None, projected=True), _player("new"))
    assert returning(before, now) == [_row(1000, 300)]


def test_a_player_on_the_table_with_no_minutes_has_not() -> None:
    before = _table(2024, _player("stays", minutes=300), _player("idle", minutes=700))
    now = _table(2025, _player("stays"), _player("idle", minutes=0))
    assert returning(before, now) == [_row(1000, 300)]


def test_a_transfer_is_gone_from_the_team_he_left() -> None:
    before = _table(
        2024,
        _player("mover", "1", 400),
        _player("stays", "1", 600),
        _player("x", "2", 500, team="Away St"),
    )
    now = _table(
        2025,
        _player("mover", "2", team="Away St"),
        _player("stays", "1"),
        _player("x", "2", team="Away St"),
    )
    assert returning(before, now) == [
        _row(1000, 600),
        TeamReturning("mens", 2025, "2", "Away St", 500, 500),
    ]


def test_a_team_missing_from_the_new_table_isnt_all_turnover() -> None:
    before = _table(2024, _player("a", "1"), _player("b", "2", team="Away St"))
    assert returning(before, _table(2025, _player("b", "2", team="Away St"))) == [
        TeamReturning("mens", 2025, "2", "Away St", 100, 100)
    ]


def test_last_seasons_projections_dont_count_as_its_minutes() -> None:
    before = _table(2024, _player("a", minutes=50, projected=True), _player("b"))
    assert returning(before, _table(2025, _player("a"), _player("b"))) == [
        _row(100, 100)
    ]


def test_every_season_needs_the_table_before_it() -> None:
    tables = [
        _table(2025, _player("a")),
        _table(2023, _player("a")),
        _table(2024, _player("a")),
    ]
    assert every_season(tables) == [_row(100, 100, 2024), _row(100, 100, 2025)]


def test_a_rebuild_uploads_the_file_and_forgets_the_old_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "mens_turnover.csv"
    uploaded: list[Path] = []
    prefixes: list[str] = []

    async def download(bucket: str, prefix: str) -> list[Path]:
        prefixes.append(prefix)
        return [Path("2024"), Path("2025")]

    async def upload(bucket: str, paths: list[Path]) -> list[str]:
        uploaded.extend(paths)
        return [str(p) for p in paths]

    tables = {
        "2024": _table(2024, _player("a", minutes=800), _player("b", minutes=200)),
        "2025": _table(2025, _player("a")),
    }
    monkeypatch.setattr(turnover_build.artifacts, "download", download)
    monkeypatch.setattr(turnover_build.artifacts, "upload", upload)
    monkeypatch.setattr(turnover_build, "read_players_file", lambda p: tables[p.name])
    monkeypatch.setattr(turnover_build, "turnover_path", lambda league: path)
    monkeypatch.setattr(
        "cassandra.predictor.turnover.turnover_path", lambda league: path
    )
    load_turnover.cache_clear()
    assert load_turnover("mens") == {}
    try:
        asyncio.run(rebuild("b", "mens"))
        assert prefixes == ["players/mens_players_"] and uploaded == [path]
        assert load_turnover("mens") == {2025: {"Home U": pytest.approx(0.2)}}
    finally:
        load_turnover.cache_clear()
