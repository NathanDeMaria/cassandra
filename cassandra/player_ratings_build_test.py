import asyncio
from datetime import date
from pathlib import Path

import pytest

from cassandra import player_ratings_build
from cassandra.player_ratings_build import inputs, refresh, season_for


def test_a_season_is_named_for_the_year_it_starts() -> None:
    assert season_for(date(2026, 11, 3)) == 2026
    assert season_for(date(2027, 3, 30)) == 2026
    assert season_for(date(2027, 8, 1)) == 2027


def test_each_input_lands_where_ten_minute_version_reads_it() -> None:
    found = inputs("mens", 2026)
    assert found["processed/basketball_plays/league=mens/season=2026/"] == (
        "plays/league=mens/season=2026/"
    )
    assert found["ten-minute-version/live/mens-2026.pkl"] == "live/mens-2026.pkl"


class _Bucket:
    """What `refresh` reads and writes, with no s3: `kit` says whether the
    season has one."""

    def __init__(self, kit: bool) -> None:
        self.kit = kit
        self.uploaded: list[Path] = []
        self.mornings: list[dict] = []
        self.turnover_rebuilt: list[str] = []

    async def copy_down(self, bucket: str, prefix: str, dest: Path) -> int:
        if prefix.endswith(".pkl") and self.kit:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"kit")
            return 1
        return 0

    async def download(self, bucket: str, prefix: str) -> list[Path]:
        return []

    async def upload(self, bucket: str, paths) -> list[str]:
        self.uploaded += list(paths)
        return [str(p) for p in paths]

    async def rebuild_turnover(self, bucket: str, league: str) -> list[str]:
        self.turnover_rebuilt.append(league)
        return []

    def morning(self, league, season, today, **paths) -> None:
        self.mornings.append(
            {"league": league, "season": season, "today": today, **paths}
        )


def _patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket: _Bucket) -> None:
    monkeypatch.setattr(player_ratings_build, "_copy_down", bucket.copy_down)
    monkeypatch.setattr(player_ratings_build.artifacts, "download", bucket.download)
    monkeypatch.setattr(player_ratings_build.artifacts, "upload", bucket.upload)
    monkeypatch.setattr(
        player_ratings_build,
        "player_ratings_path",
        lambda league: tmp_path / f"{league}_player_ratings.csv",
    )
    monkeypatch.setattr(
        player_ratings_build, "luck_path", lambda league: tmp_path / "no_luck.csv"
    )
    monkeypatch.setattr(
        player_ratings_build,
        "players_csv_path",
        lambda league, season: tmp_path / "players" / f"{league}_players_{season}.csv",
    )
    monkeypatch.setattr(
        player_ratings_build,
        "roster_path",
        lambda league, season: tmp_path
        / "players"
        / f"{league}_{season}_roster.parquet",
    )
    monkeypatch.setattr(
        player_ratings_build, "rebuild_turnover", bucket.rebuild_turnover
    )
    monkeypatch.setattr("ten_minute_version.build.morning", bucket.morning)


def test_a_season_without_a_kit_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bucket = _Bucket(kit=False)
    _patch(monkeypatch, tmp_path, bucket)
    assert not asyncio.run(refresh("b", "mens", date(2027, 6, 1)))
    assert bucket.mornings == [] and bucket.uploaded == []
    assert bucket.turnover_rebuilt == []


def test_a_morning_runs_on_the_copies_and_uploads_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bucket = _Bucket(kit=True)
    _patch(monkeypatch, tmp_path, bucket)
    assert asyncio.run(refresh("b", "womens", date(2026, 11, 10)))
    (run,) = bucket.mornings
    assert (run["league"], run["season"], run["today"]) == (
        "womens",
        2026,
        "2026-11-10",
    )
    assert run["kit"].endswith("/live") and run["out"] == str(tmp_path)
    assert bucket.uploaded == [tmp_path / "womens_player_ratings.csv"]
    assert bucket.turnover_rebuilt == ["womens"]


def test_a_morning_keeps_its_table_and_roster_for_the_next_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bucket = _Bucket(kit=True)
    _patch(monkeypatch, tmp_path, bucket)
    kept = tmp_path / "players" / "mens_2026_roster.parquet"
    kept.parent.mkdir()
    kept.write_bytes(b"last roster")
    seen: list[bytes] = []

    async def download(bucket_name: str, prefix: str) -> list[Path]:
        return [kept] if prefix == "players/mens_2026_roster.parquet" else []

    def morning(league, season, today, **paths) -> None:
        tables = Path(paths["tables"]) / "league=mens" / "season=2026"
        seen.append((tables / "roster.parquet").read_bytes())
        (tables / "roster.parquet").write_bytes(b"this morning's")
        (Path(paths["out"]) / "mens_players_2026.csv").write_text("table")

    monkeypatch.setattr(player_ratings_build.artifacts, "download", download)
    monkeypatch.setattr("ten_minute_version.build.morning", morning)
    assert asyncio.run(refresh("b", "mens", date(2026, 11, 10)))
    table = tmp_path / "players" / "mens_players_2026.csv"
    assert seen == [b"last roster"]
    assert kept.read_bytes() == b"this morning's" and table.read_text() == "table"
    assert not (tmp_path / "mens_players_2026.csv").exists()
    assert set(bucket.uploaded) == {table, kept, tmp_path / "mens_player_ratings.csv"}
