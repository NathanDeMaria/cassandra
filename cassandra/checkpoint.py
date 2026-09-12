"""Where a search saves itself, so a reclaimed container resumes instead of restarting.

The optimize queue is all spot, and Batch retries a reclaimed job from a
cold container. Before this, an attempt that died three hours in had
produced nothing: `jobs/main.tf` raised the retry count to six because of
it, and said plainly that the exposure "is not a number that gets safer as
more `n_iter` goes in". A thirteen-knob search at a thousand probes on
ncaafb is several hours of replay, and on 2026-09-12 one such child was
reclaimed twice and restarted from zero each time.

`bayes_opt` can serialize a search whole -- `save_state` and `load_state`
carry the probes, the fitted process, the acquisition's own counters and
the random state -- and a chunked `maximize` reproduces an unchunked one
probe for probe (`cassandra/optimize_test.py` holds that, with a reload
between every chunk). So the optimizer saves after every chunk and on a
SIGTERM, and a retry loads the save and runs the probes that are left.

**Keyed by the Batch job, not by the config.** A retry is the same job id
with a higher `AWS_BATCH_JOB_ATTEMPT`, so `for_this_job` finds the save the
previous attempt left. A later run of the same config is a different job
and starts clean, which is the point: it is searching different seasons
and a rebuilt index, and a save from last week would resume a search over
data that no longer exists. The job deletes its save on success, and the
prefix carries a lifecycle rule for the ones a job that spent every attempt
leaves behind -- see `CHECKPOINT_PREFIX`.

Outside Batch there is no job id and nothing is saved, unless a caller
hands the optimizer a `FileCheckpoint` -- which is what the tests do, and
what a long local search can do to survive a closed laptop.
"""

import asyncio
import os
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, Self

from aiobotocore.session import get_session
from botocore.exceptions import ClientError

# Under the shared bucket's `cassandra/` prefix like every other artifact,
# in a directory of its own so a lifecycle rule can expire it: nothing here
# is worth keeping past the job that wrote it, and a job that fails every
# attempt leaves its last save behind. Apply on the bucket, once:
#
#   {"Rules": [{"ID": "cassandra-checkpoints", "Status": "Enabled",
#               "Filter": {"Prefix": "cassandra/checkpoints/"},
#               "Expiration": {"Days": 7}}]}
#
# Seven days is longer than any job can live (Batch's own timeout is hours),
# so a rule can never expire a save an attempt is about to resume from.
CHECKPOINT_PREFIX = "cassandra/checkpoints/"

# How Batch tells a container which job it is. An array child's id is
# `<parent>:<index>`, so the two together name one search.
_JOB_ID_VAR = "AWS_BATCH_JOB_ID"


class Checkpoint(Protocol):
    """Somewhere a search's serialized state can be put, fetched and dropped."""

    def load(self) -> bytes | None:
        """The last save, or None if there isn't one."""

    def save(self, state: bytes) -> None: ...

    def clear(self) -> None:
        """Drop the save. Called once the search has finished and its result
        is safe; a no-op if there was nothing to drop."""


class FileCheckpoint:
    """A save on local disk, for tests and for a long search on a laptop."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> bytes | None:
        return self._path.read_bytes() if self._path.is_file() else None

    def save(self, state: bytes) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write beside and rename, so a SIGKILL mid-write leaves the previous
        # save rather than half of this one.
        partial = self._path.with_suffix(self._path.suffix + ".partial")
        partial.write_bytes(state)
        partial.replace(self._path)

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)


class S3Checkpoint:
    """A save in the batch bucket, under `CHECKPOINT_PREFIX`.

    Synchronous on purpose: `bayes_opt` drives the search from a plain loop,
    and the SIGTERM handler that makes the save worth having runs on the
    main thread with no event loop to await on. Each call runs its own
    small loop on a worker thread, which is a few milliseconds of overhead
    against a put that is itself a network round trip.
    """

    def __init__(self, bucket: str, key: str) -> None:
        self._bucket = bucket
        self._key = key

    @classmethod
    def for_this_job(cls, bucket: str) -> Self | None:
        """The save slot for the Batch job this process is, or None outside Batch."""
        job_id = os.environ.get(_JOB_ID_VAR)
        if not job_id:
            return None
        return cls(bucket, f"{CHECKPOINT_PREFIX}{job_id.replace(':', '-')}.json")

    @property
    def key(self) -> str:
        return self._key

    def load(self) -> bytes | None:
        async def get() -> bytes | None:
            async with get_session().create_client("s3") as client:
                try:
                    response = await client.get_object(
                        Bucket=self._bucket, Key=self._key
                    )
                except ClientError as error:
                    if error.response["Error"]["Code"] in ("NoSuchKey", "404"):
                        return None
                    raise
                async with response["Body"] as stream:
                    return await stream.read()

        return _run(get())

    def save(self, state: bytes) -> None:
        async def put() -> None:
            async with get_session().create_client("s3") as client:
                await client.put_object(Bucket=self._bucket, Key=self._key, Body=state)

        _run(put())

    def clear(self) -> None:
        """Drop the save; a failure here is logged, not raised.

        Housekeeping after a search that has already produced its result,
        so a denied delete -- a role without the grant, say -- must not turn
        a finished search into a failed job. The lifecycle rule on the prefix
        picks up what this leaves.
        """

        async def delete() -> None:
            async with get_session().create_client("s3") as client:
                await client.delete_object(Bucket=self._bucket, Key=self._key)

        try:
            _run(delete())
        except ClientError as error:
            print(f"[optimize] could not delete the checkpoint {self._key}: {error}")


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run one coroutine to completion from synchronous code.

    On its own thread rather than `asyncio.run` here, because the caller may
    already be inside a running loop -- `optimize.py`'s entry point is async
    and calls the synchronous search from it -- and `asyncio.run` refuses to
    nest.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        # A lambda rather than `asyncio.run` itself, so the type of what comes
        # back is the coroutine's own rather than `asyncio.run`'s generic.
        return pool.submit(lambda: asyncio.run(coroutine)).result()
