"""`download_predictor_data` has to drop every cached read, not most of them.

The failure this guards against is silent: `load_manifest` builds a predictor
per config before the download, each loader caches the empty read it makes
then, and the file that lands a moment later is never opened. The fit is
different and nothing fails. It happened to the quarterback index for its
first month -- see the comment in `download_predictor_data`.
"""

import asyncio
from pathlib import Path

import pytest

from cassandra.predictor.base_predictor import load_anchors
from cassandra.predictor.epa import load_epa
from cassandra.predictor.game_control import load_game_control
from cassandra.predictor.qb_out import load_qb_out

from . import artifacts

_LEAGUE = "nfl"

#: Every cached reader of a file under `predictor/data`. A new loader goes
#: here as well as in `download_predictor_data`, and the test below is what
#: notices if it only made it into one of the two.
_LOADERS = (load_anchors, load_game_control, load_epa, load_qb_out)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the quarterback index at tmp_path, and start every cache empty."""
    monkeypatch.setattr(
        "cassandra.predictor.qb_out.qb_out_path",
        lambda league: tmp_path / f"{league}_qb_out.json",
    )
    for loader in _LOADERS:
        loader.cache_clear()
    yield
    for loader in _LOADERS:
        loader.cache_clear()


def test_the_download_drops_every_cached_read(monkeypatch: pytest.MonkeyPatch) -> None:
    for loader in _LOADERS:
        loader(_LEAGUE)
    assert all(loader.cache_info().currsize == 1 for loader in _LOADERS)

    async def nothing_to_download(bucket: str, prefix: str = "models/") -> list[Path]:
        return []

    monkeypatch.setattr(artifacts, "download", nothing_to_download)
    asyncio.run(artifacts.download_predictor_data("bucket"))

    assert [loader.cache_info().currsize for loader in _LOADERS] == [0, 0, 0, 0]


def test_the_index_written_by_the_download_is_what_the_next_read_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The order the bug needs: a read before the file exists, cached empty.
    assert load_qb_out(_LEAGUE) == {}

    async def fake_download(bucket: str, prefix: str = "models/") -> list[Path]:
        path = tmp_path / f"{_LEAGUE}_qb_out.json"
        path.write_text('{"league": "nfl", "games": {"g1": ["Chicago Bears"]}}')
        return [path]

    monkeypatch.setattr(artifacts, "download", fake_download)
    asyncio.run(artifacts.download_predictor_data("bucket"))

    assert load_qb_out(_LEAGUE) == {"g1": frozenset({"Chicago Bears"})}
