"""A search that saves itself resumes where it was, and finishes the same.

The property everything rests on is the first test: chunked probing with a
reload between every chunk reproduces one unbroken `maximize` probe for
probe. The rest is the plumbing -- that an interrupted search picks up from
its save rather than from zero, that a SIGTERM makes a save, that a
finished search drops its save, and that a Batch job finds its own slot.
"""

import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest

from .checkpoint import CHECKPOINT_PREFIX, FileCheckpoint, S3Checkpoint
from .constants import temp_bucket
from .optimize import INIT_POINTS, optimize

BOUNDS = {"x": (-3.0, 3.0), "y": (-3.0, 3.0)}


def _bowl(x: float, y: float) -> float:
    return -((x - 1.3) ** 2) - (y + 0.4) ** 2


class _Recording:
    """The objective, remembering every probe so two searches can be compared."""

    def __init__(self, fail_at: int | None = None, signal_at: int | None = None):
        self.probes: list[tuple[float, float]] = []
        self.fail_at = fail_at
        self.signal_at = signal_at

    def __call__(self, x: float, y: float) -> float:
        self.probes.append((x, y))
        if self.fail_at is not None and len(self.probes) == self.fail_at:
            raise RuntimeError("reclaimed")
        if self.signal_at is not None and len(self.probes) == self.signal_at:
            os.kill(os.getpid(), signal.SIGTERM)
        return _bowl(x, y)


def test_chunked_with_reloads_reproduces_one_unbroken_search(tmp_path: Path) -> None:
    """Probe for probe, not just in the answer.

    A three-probe chunk against a thirty-probe search is a reload every few
    steps, which is more than any retry does; if the random state or the
    acquisition's counters were not in the save, the sequences would part
    company at the first chunk boundary.
    """
    unbroken = _Recording()
    optimize(unbroken, BOUNDS, iterations=20)

    reloaded = _Recording()
    result = optimize(
        reloaded,
        BOUNDS,
        iterations=20,
        checkpoint=FileCheckpoint(tmp_path / "search.json"),
        checkpoint_every=3,
    )

    assert reloaded.probes == unbroken.probes
    assert result == optimize(_Recording(), BOUNDS, iterations=20)


def test_an_interrupted_search_resumes_from_its_save(tmp_path: Path) -> None:
    """The retry: same checkpoint, a fresh optimizer, the probes that are left."""
    unbroken = _Recording()
    optimize(unbroken, BOUNDS, iterations=20)
    checkpoint = FileCheckpoint(tmp_path / "search.json")

    first = _Recording(fail_at=17)
    with pytest.raises(RuntimeError, match="reclaimed"):
        optimize(
            first, BOUNDS, iterations=20, checkpoint=checkpoint, checkpoint_every=5
        )
    # Saved after every fifth probe, so the last save holds fifteen and the
    # two probes after it are the ones a retry pays for again.
    assert checkpoint.load() is not None

    second = _Recording()
    optimize(second, BOUNDS, iterations=20, checkpoint=checkpoint, checkpoint_every=5)

    total = INIT_POINTS + 20
    assert len(second.probes) == total - 15
    assert first.probes[:15] + second.probes == unbroken.probes


def test_a_sigterm_saves_before_stopping(tmp_path: Path) -> None:
    """The reclaim notice, which lands between periodic saves.

    The save the handler makes holds every probe scored so far, so the
    retry loses nothing -- against the chunk it would otherwise lose.
    """
    checkpoint = FileCheckpoint(tmp_path / "search.json")
    interrupted = _Recording(signal_at=12)
    with pytest.raises(SystemExit) as stopped:
        optimize(
            interrupted,
            BOUNDS,
            iterations=20,
            checkpoint=checkpoint,
            checkpoint_every=25,
        )
    assert stopped.value.code == 128 + signal.SIGTERM
    # The default disposition is back, so a later SIGTERM in this process
    # would not try to save a search that no longer exists.
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    resumed = _Recording()
    optimize(resumed, BOUNDS, iterations=20, checkpoint=checkpoint, checkpoint_every=25)

    unbroken = _Recording()
    optimize(unbroken, BOUNDS, iterations=20)
    # Twelve scored before the signal; the twelfth is the one in flight, whose
    # result never reached the optimizer, so the save holds eleven and the
    # retry pays for it again.
    resumed_path = interrupted.probes[:11] + resumed.probes
    assert len(resumed_path) == len(unbroken.probes)
    # Approximately, unlike the chunk-boundary reloads above, which are
    # exact: a save taken mid-probe carries a random state already advanced
    # for the suggestion in flight, so the resumed acquisition search starts
    # from different seeds and lands on the same optimum to ~1e-6 -- and
    # those differences compound to ~1e-4 by the twentieth probe after. The
    # same search, not a different one; a retry that lands a chunk boundary
    # off gets the exact one.
    assert np.allclose(resumed_path, unbroken.probes, atol=1e-3)


def test_a_finished_search_drops_its_save(tmp_path: Path) -> None:
    checkpoint = FileCheckpoint(tmp_path / "search.json")
    optimize(
        _Recording(), BOUNDS, iterations=10, checkpoint=checkpoint, checkpoint_every=4
    )

    assert checkpoint.load() is None
    assert not (tmp_path / "search.json").exists()


def test_a_search_without_a_checkpoint_is_untouched(tmp_path: Path) -> None:
    """No save, no signal handler, same answer."""
    before = signal.getsignal(signal.SIGTERM)
    plain = _Recording()
    optimize(plain, BOUNDS, iterations=10)

    assert signal.getsignal(signal.SIGTERM) is before
    assert not list(tmp_path.iterdir())


def test_a_file_save_replaces_rather_than_truncates(tmp_path: Path) -> None:
    checkpoint = FileCheckpoint(tmp_path / "deep" / "search.json")
    checkpoint.save(b"one")
    checkpoint.save(b"two")

    assert checkpoint.load() == b"two"
    assert not (tmp_path / "deep" / "search.json.partial").exists()
    checkpoint.clear()
    checkpoint.clear()  # nothing to drop is not an error
    assert checkpoint.load() is None


def test_a_batch_job_finds_its_own_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keyed by the job id, with an array child's colon made key-safe, in the
    temp bucket."""
    monkeypatch.setenv("CASSANDRA_TEMP_BUCKET", "scratch")
    monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)
    assert S3Checkpoint.for_this_job() is None

    monkeypatch.setenv("AWS_BATCH_JOB_ID", "2d995432-63be:7")
    slot = S3Checkpoint.for_this_job()
    assert slot is not None
    assert slot.bucket == "scratch"
    assert slot.key == f"{CHECKPOINT_PREFIX}2d995432-63be-7.json"


def test_the_temp_bucket_comes_from_the_environment_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = tmp_path / "config.json"
    outputs.write_text(json.dumps({"temp_bucket": {"value": "from-file"}}))
    monkeypatch.setattr("cassandra.constants._OUTPUTS_FILE", outputs)

    monkeypatch.setenv("CASSANDRA_TEMP_BUCKET", "from-env")
    assert temp_bucket() == "from-env"

    monkeypatch.delenv("CASSANDRA_TEMP_BUCKET")
    assert temp_bucket() == "from-file"

    monkeypatch.setattr("cassandra.constants._OUTPUTS_FILE", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="CASSANDRA_TEMP_BUCKET"):
        temp_bucket()
