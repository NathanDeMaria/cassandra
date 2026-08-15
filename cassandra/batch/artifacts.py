"""Moving `~/.cassandra` in and out of s3, because containers don't keep it.

Everything the pipeline generates -- optimizer results, predictor state,
calibrations, evaluation csvs -- lands under `CASSANDRA_HOME`, which on a
laptop persists between runs and in a Batch container evaporates when the job
exits. Splitting optimize from publish means the second job runs on a machine
that never saw the first one's disk, so the handoff has to go through s3.

The s3 layout mirrors the local one exactly, rooted at `ARTIFACT_PREFIX`:
`~/.cassandra/models/mens/elo_result.json` is
`s3://<bucket>/cassandra/models/mens/elo_result.json`. Keeping the two
identical means a key is readable without a decoder, and `download` can write
straight to the path the rest of the code already expects to find things at.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from aiobotocore.session import get_session

from cassandra.constants import CASSANDRA_HOME

# Shared bucket, so cassandra's generated files get a prefix of their own
# rather than sitting next to endgame's `seasons/` and `odds/`.
ARTIFACT_PREFIX = "cassandra"

# S3 has no directories, so a "download the results" call is a prefix listing.
# Uploads go one object at a time but concurrently; these are small json files
# and the round trip dominates.
_MAX_CONCURRENCY = 16


def _key(path: Path) -> str:
    """The s3 key for a path under CASSANDRA_HOME."""
    relative = path.resolve().relative_to(CASSANDRA_HOME.resolve())
    return f"{ARTIFACT_PREFIX}/{relative.as_posix()}"


def _path(key: str) -> Path:
    """The local path for an s3 key, the inverse of `_key`."""
    return CASSANDRA_HOME / Path(key).relative_to(ARTIFACT_PREFIX)


async def upload(bucket: str, paths: Iterable[Path]) -> list[str]:
    """Put files from under CASSANDRA_HOME into s3, returning the keys written.

    Missing paths are skipped rather than raised on: a predictor that builds no
    priors, or an optimize run whose model produced no state file, is ordinary
    and shouldn't fail the upload that follows it.
    """
    present = [path for path in paths if path.is_file()]
    if not present:
        return []

    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    session = get_session()
    async with session.create_client("s3") as client:

        async def put(path: Path) -> str:
            key = _key(path)
            async with semaphore:
                await client.put_object(Bucket=bucket, Key=key, Body=path.read_bytes())
            return key

        return await asyncio.gather(*(put(path) for path in present))


async def download(bucket: str, prefix: str = "models/") -> list[Path]:
    """Pull everything under a CASSANDRA_HOME-relative prefix onto local disk.

    Returns the paths written. An empty list means the prefix is empty, which
    callers generally want to treat as an error with a better message than a
    downstream `FileNotFoundError` -- optimize hasn't run, or ran against a
    different bucket.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    session = get_session()
    async with session.create_client("s3") as client:
        keys = [
            key
            async for key in _list_keys(client, bucket, f"{ARTIFACT_PREFIX}/{prefix}")
        ]

        async def get(key: str) -> Path:
            async with semaphore:
                response = await client.get_object(Bucket=bucket, Key=key)
                async with response["Body"] as stream:
                    body = await stream.read()
            path = _path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            return path

        return await asyncio.gather(*(get(key) for key in keys))


async def _list_keys(client, bucket: str, prefix: str) -> AsyncIterator[str]:
    paginator = client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            # A key ending in "/" is a console-created folder marker, not a
            # file, and writing it locally would mean creating a directory
            # where a later key wants a file.
            if not item["Key"].endswith("/"):
                yield item["Key"]


def results_for(league: str, model: str) -> list[Path]:
    """The files one optimize run produces, whether or not they all exist.

    `upload` skips the ones that don't, so this can name everything a run
    *might* write without the caller checking first.
    """
    league_dir = CASSANDRA_HOME / "models" / league
    return [
        league_dir / f"{model}_result.json",
        league_dir / f"{model}_state.json",
    ]
