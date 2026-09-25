import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from . import replay_cache
from .odds import OddsDatabase, OddsSnapshot


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("AUTHORED_DIR", "GENERATED_DIR", "REPLAY_DIR"):
        monkeypatch.setattr(replay_cache, name, tmp_path / name.lower())
    (tmp_path / "authored_dir" / "nfl").mkdir(parents=True)
    (tmp_path / "authored_dir" / "nfl" / "m.json").write_text('{"fixed": {"a": 1}}')
    return tmp_path


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "date": pd.to_datetime(["2026-09-06", "2026-09-13"], utc=True),
            "spread": [-3.0, None],
            "team1_win_prob": [0.6, 0.4],
        }
    )


@pytest.fixture
def replays(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    async def fake(config, league, state_path, priors_from=None, priors_path=None):
        # Never the shared priors under ~/.cassandra/predictor/data.
        assert priors_path is not None
        assert priors_path.parent == replay_cache.REPLAY_DIR / league
        calls.append(league)
        return _frame()

    monkeypatch.setattr(replay_cache, "get_predictions", fake)
    return calls


def test_a_generated_result_wins_over_the_baseline(dirs: Path) -> None:
    assert replay_cache.config_path("nfl", "m") == dirs / "authored_dir/nfl/m.json"
    generated = dirs / "generated_dir/nfl/m_result.json"
    generated.parent.mkdir(parents=True)
    generated.write_text("{}")
    assert replay_cache.config_path("nfl", "m") == generated
    with pytest.raises(FileNotFoundError):
        replay_cache.config_path("nfl", "nothing")


def test_the_key_follows_the_config_by_content(dirs: Path) -> None:
    before = replay_cache.replay_key("nfl", "m")
    config = dirs / "authored_dir/nfl/m.json"
    config.write_text('{"fixed": {"a": 1}}')
    assert replay_cache.replay_key("nfl", "m") == before
    config.write_text('{"fixed": {"a": 2}}')
    assert replay_cache.replay_key("nfl", "m") != before


def test_the_key_reads_the_replay_s_code_and_not_its_readers() -> None:
    sources = {
        p.relative_to(Path(replay_cache.__file__).parent).as_posix()
        for p in replay_cache._replay_sources()
    }
    assert "predictor/glicko.py" in sources
    assert "save_predictions.py" in sources
    for reader in ("team_seasons.py", "evidence.py", "betting.py", "residuals.py"):
        assert reader not in sources
    assert not any(s.endswith("_test.py") for s in sources)


def test_a_current_replay_is_read_back_and_a_stale_one_is_made_again(
    dirs: Path, replays: list[str]
) -> None:
    first = asyncio.run(replay_cache.load_replay("nfl", "m"))
    assert not first.from_cache and replays == ["nfl"]
    second = asyncio.run(replay_cache.load_replay("nfl", "m"))
    assert second.from_cache and replays == ["nfl"]
    pd.testing.assert_frame_equal(
        second.predictions, first.predictions.assign(spread=[-3.0, float("nan")])
    )
    # Too old, then a changed config, then asked for: each replays.
    meta = dirs / "replay_dir/nfl/m.json"
    stamp = json.loads(meta.read_text())
    stamp["replayed_at"] = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    meta.write_text(json.dumps(stamp))
    assert not asyncio.run(replay_cache.load_replay("nfl", "m")).from_cache
    (dirs / "authored_dir/nfl/m.json").write_text('{"fixed": {"a": 3}}')
    assert not asyncio.run(replay_cache.load_replay("nfl", "m")).from_cache
    assert not asyncio.run(
        replay_cache.load_replay("nfl", "m", refresh=True)
    ).from_cache
    assert len(replays) == 4


def test_a_broken_cache_is_a_miss(dirs: Path, replays: list[str]) -> None:
    asyncio.run(replay_cache.load_replay("nfl", "m"))
    (dirs / "replay_dir/nfl/m.parquet").write_text("not parquet")
    assert not asyncio.run(replay_cache.load_replay("nfl", "m")).from_cache


def test_the_odds_history_survives_the_round_trip() -> None:
    at = datetime(2026, 9, 12, 15, 0, tzinfo=UTC)
    history = {
        "g": [
            OddsSnapshot(at, -3.0, -150.0, 130.0),
            OddsSnapshot(at + timedelta(hours=1), None, None, 120.0),
        ],
        "h": [OddsSnapshot(at, 7.0, None, None)],
    }
    frame = replay_cache._odds_to_frame(OddsDatabase.from_history(history))
    back = replay_cache._odds_from_frame(frame)
    assert back.snapshots("g") == history["g"]
    assert back.snapshots("h") == history["h"]
    odds = back.get_odds("g")
    assert odds is not None and odds.spread == -3.0
