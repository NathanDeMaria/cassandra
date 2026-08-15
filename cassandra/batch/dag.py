"""Submitting the run as a DAG of Batch jobs.

AWS Batch puts job *dependencies* on `SubmitJob`, not on the job definition,
so terraform can declare the nodes but not the edges. This is the edges.

The shape:

    optimize (array job, one child per league/model)
        |
        +--> evaluate (one job, all leagues)
        |
        +--> publish  (array job, one child per league)

evaluate and publish are siblings, not a chain: `publish.py` reads
`<model>_result.json` and fits its own prob->margin mapping via `_best_fit`,
so it needs the optimizer's output but nothing evaluate produces. Making it
wait would cost it the length of an evaluation run for no input.

A dependency on an array job's *parent* waits for every child, which is what
both downstream stages want, and it costs one entry against Batch's limit of
20 dependencies per job rather than one per model.
"""

import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone

from aiobotocore.session import get_session

from cassandra.batch import manifest
from cassandra.batch.manifest import Work

# AWS Batch rejects an array job of size 1 -- `arrayProperties.size` has to be
# at least 2. A single-model run is the normal way to test a change, so it
# falls back to a plain job with the index pinned in the environment.
_MIN_ARRAY_SIZE = 2


@dataclass(frozen=True)
class Submitted:
    """What went to Batch, for printing and for wiring up dependencies."""

    name: str
    job_id: str
    size: int


def _job_name(stem: str) -> str:
    """Batch job names have to be unique-ish and match [A-Za-z0-9_-]."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"cassandra-{stem}-{stamp}"


async def submit(
    *,
    optimize_job_definition: str,
    evaluate_job_definition: str,
    publish_job_definition: str,
    job_queue: str,
    leagues: list[str] | None = None,
    models: list[str] | None = None,
    skip_optimize: bool = False,
    skip_evaluate: bool = False,
    skip_publish: bool = False,
    dry_run: bool = False,
) -> list[Submitted]:
    """Submit the DAG, returning what was submitted in dependency order.

    `skip_optimize` is what the daily publish schedule uses: republish from
    whatever results are already in s3, without paying for a re-search.
    """
    work = manifest.load_manifest(leagues=leagues, models=models)
    if not work and not skip_optimize:
        raise ValueError(
            "No optimization configs matched"
            + (f" leagues={leagues}" if leagues else "")
            + (f" models={models}" if models else "")
        )

    # Publish fans out over leagues, not models: `publish.py --league` reads
    # that league's seasons and odds once and does every model in it, so a
    # child per model would re-read the same s3 objects per model.
    publish_leagues = sorted(leagues) if leagues else sorted({w.league for w in work})

    # A dry run builds every request and sends none, so it must not need a
    # client -- creating one costs a region and credentials, which is exactly
    # what someone checking "what would this submit?" hasn't set up yet.
    async with AsyncExitStack() as stack:
        client = (
            None
            if dry_run
            else await stack.enter_async_context(get_session().create_client("batch"))
        )
        submitter = _Submitter(client, job_queue, dry_run)
        submitted: list[Submitted] = []

        optimize_job = None
        if not skip_optimize:
            optimize_job = await submitter.run(
                name=_job_name("optimize"),
                job_definition=optimize_job_definition,
                command=["optimize"],
                size=len(work),
                # Children resolve their index against this exact list rather
                # than rebuilding it, so a launcher and an image on different
                # commits fail loudly instead of quietly disagreeing.
                environment={manifest.MANIFEST_ENV_VAR: manifest.encode(work)},
            )
            submitted.append(optimize_job)

        # Both downstream stages depend on the whole array. With
        # --skip-optimize there's nothing to wait for and they start at once.
        depends_on = [optimize_job.job_id] if optimize_job else []

        if not skip_evaluate:
            submitted.append(
                await submitter.run(
                    name=_job_name("evaluate"),
                    job_definition=evaluate_job_definition,
                    command=["evaluate"] + _league_args(leagues),
                    size=1,
                    depends_on=depends_on,
                )
            )

        if not skip_publish:
            submitted.append(
                await submitter.run(
                    name=_job_name("publish"),
                    job_definition=publish_job_definition,
                    command=["publish"],
                    size=len(publish_leagues),
                    environment={_PUBLISH_LEAGUES_ENV_VAR: ",".join(publish_leagues)},
                    depends_on=depends_on,
                )
            )

    return submitted


# The publish array child reads its league out of this by array index, the
# same trick the optimize array uses for its work list.
_PUBLISH_LEAGUES_ENV_VAR = "CASSANDRA_PUBLISH_LEAGUES"


def _league_args(leagues: list[str] | None) -> list[str]:
    return [arg for league in leagues or [] for arg in ("--league", league)]


class _Submitter:
    """One `submit_job` call, with the array/plain and dry-run branches."""

    def __init__(self, client, job_queue: str, dry_run: bool) -> None:
        self._client = client
        self._job_queue = job_queue
        self._dry_run = dry_run

    async def run(
        self,
        *,
        name: str,
        job_definition: str,
        command: list[str],
        size: int,
        environment: dict[str, str] | None = None,
        depends_on: list[str] | None = None,
    ) -> Submitted:
        request: dict = {
            "jobName": name,
            "jobQueue": self._job_queue,
            "jobDefinition": job_definition,
            "containerOverrides": {
                "command": command,
                "environment": [
                    {"name": key, "value": value}
                    for key, value in (environment or {}).items()
                ],
            },
        }
        if size >= _MIN_ARRAY_SIZE:
            request["arrayProperties"] = {"size": size}
        else:
            # A one-item "array" runs as a plain job, so nothing sets
            # AWS_BATCH_JOB_ARRAY_INDEX and the child would have no way to
            # know which item it is. Pin it to 0.
            request["containerOverrides"]["environment"].append(
                {"name": manifest.ARRAY_INDEX_ENV_VAR, "value": "0"}
            )
        if depends_on:
            request["dependsOn"] = [{"jobId": job_id} for job_id in depends_on]

        if self._dry_run:
            return Submitted(name=name, job_id=f"dry-run-{name}", size=size)

        response = await self._client.submit_job(**request)
        return Submitted(name=name, job_id=response["jobId"], size=size)


async def wait(job_ids: list[str], poll_seconds: int = 30) -> dict[str, str]:
    """Poll until every job reaches a terminal state; returns id -> status.

    Only for interactive use (`make submit-wait`). The scheduled launcher
    submits and exits -- holding a container open for the hours an
    optimization takes would bill for a process whose entire job is sleeping,
    and Batch's own dependency handling is what actually orders the run.
    """
    terminal = {"SUCCEEDED", "FAILED"}
    session = get_session()
    async with session.create_client("batch") as client:
        statuses: dict[str, str] = {}
        pending = list(job_ids)
        while pending:
            response = await client.describe_jobs(jobs=pending)
            for job in response["jobs"]:
                if job["status"] in terminal:
                    statuses[job["jobId"]] = job["status"]
            pending = [job_id for job_id in pending if job_id not in statuses]
            if pending:
                await asyncio.sleep(poll_seconds)
        return statuses


def publish_league(index: int) -> str:
    """Which league this publish array child owns."""
    raw = os.environ.get(_PUBLISH_LEAGUES_ENV_VAR)
    if raw is None:
        raise ValueError(
            f"{_PUBLISH_LEAGUES_ENV_VAR} is not set; the publish job expects the "
            "launcher to pin the league list it sized the array against."
        )
    leagues = raw.split(",")
    if not 0 <= index < len(leagues):
        raise IndexError(f"Array index {index} is out of range for leagues {leagues}")
    return leagues[index]


def work_for(index: int | None = None) -> Work:
    """Re-exported so the CLI has one place to ask about array indices."""
    return manifest.resolve_index(index)
