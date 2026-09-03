"""Submitting the run as a DAG of Batch jobs.

AWS Batch puts job *dependencies* on `SubmitJob`, not on the job definition,
so terraform can declare the nodes but not the edges. This is the edges.

The shape:

    anchors (array, one child per league with division anchors)
                                    |
                                    v
                        optimize (array, per model)
                                    |
                     +--------------+
                     |              |
                     v              v
         evaluate (one job)    publish (array)

anchors comes before optimize because it decides what a search is fit
against: the anchor sets the scale a rating sits on, and a search run before
it is fit against a different scale than the one the run publishes on.

There was a `game_control` sibling here, sweeping play-by-play into the index
`ControlGlickoPredictor` blends. Nothing in the run reads that index any more
-- see `cassandra.predictor.control` for what the searches found -- so the
node is gone and the sweep is a `jobs.py game_control` someone runs by hand
when there is a signal worth measuring again.

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
from cassandra.predictor import ANCHOR_LEAGUES

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
    anchors_job_definition: str,
    optimize_job_definition: str,
    evaluate_job_definition: str,
    publish_job_definition: str,
    job_queue: str,
    leagues: list[str] | None = None,
    models: list[str] | None = None,
    skip_anchors: bool = False,
    rebuild_anchors: bool = False,
    skip_optimize: bool = False,
    skip_evaluate: bool = False,
    skip_publish: bool = False,
    dry_run: bool = False,
) -> list[Submitted]:
    """Submit the DAG, returning what was submitted in dependency order.

    `skip_optimize` is what the daily publish schedule uses: republish from
    whatever results are already in s3, without paying for a re-search. It
    implies `skip_anchors`: the anchors set the scale a *search* is fit
    against, and a republish reads that scale back out of s3 rather than
    deciding it. `skip_anchors` on its own is for the run that knows its
    anchors are already built and doesn't want to pay a container to confirm
    it.

    `rebuild_anchors` is the opposite end: refit them even though s3 already
    has some, and overwrite what's there. Every rating downstream is on the
    scale the anchors set, so a run that rebuilds them re-rates the whole
    league -- which is why the anchors job defaults to leaving an existing
    file alone and this has to be asked for by name.

    Asking for it alongside anything that drops the anchors job raises rather
    than quietly doing nothing: the two readings of `--rebuild-anchors
    --skip-optimize` are "rebuild them" and "don't", and a run that re-rates
    a league is not one to guess about.
    """
    if rebuild_anchors and (skip_anchors or skip_optimize):
        dropped = "skip_anchors" if skip_anchors else "skip_optimize"
        raise ValueError(
            f"rebuild_anchors asks for an anchors job that {dropped} removes. "
            "Drop one of them."
        )
    work = manifest.load_manifest(leagues=leagues, models=models)
    if not work and not skip_optimize:
        raise ValueError(
            "No optimization configs matched"
            + (f" leagues={leagues}" if leagues else "")
            + (f" models={models}" if models else "")
        )

    # The leagues this run touches. Both fan-outs below go over leagues rather
    # than models: `publish.py --league` reads that league's seasons and odds
    # once and does every model in it, so a child per model would re-read the
    # same s3 objects per model, and an anchor fit has no per-model input at
    # all.
    scope_leagues = sorted(leagues) if leagues else sorted({w.league for w in work})

    # Only the leagues that have anchors to fit. `nfl` never does, and a run
    # scoped to it should not submit an array of zero.
    anchor_leagues = [league for league in ANCHOR_LEAGUES if league in scope_leagues]

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

        anchors_job = None
        if anchor_leagues and not (skip_anchors or skip_optimize):
            anchors_job = await submitter.run(
                name=_job_name("anchors"),
                job_definition=anchors_job_definition,
                # `--if-missing` is the anchors job's own default and means
                # "leave what s3 already has alone"; turning it off is what
                # makes the job refit and overwrite.
                command=["anchors"]
                + (["--if-missing=False"] if rebuild_anchors else []),
                size=len(anchor_leagues),
                environment={_ANCHOR_LEAGUES_ENV_VAR: ",".join(anchor_leagues)},
            )
            submitted.append(anchors_job)

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
                # Every child is fit against the scale the anchors set.
                # Waiting on an array's parent waits for all of it, so no
                # child starts searching against a scale still being fit.
                depends_on=[anchors_job.job_id] if anchors_job else [],
            )
            submitted.append(optimize_job)

        # Both downstream stages wait on the optimize array, and through it on
        # anchors -- there is no anchors job without an optimize job to feed,
        # so naming it here as well would be an edge that can never be the
        # only one. A --skip-optimize republish has nothing upstream left to
        # wait for, and starts immediately.
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
                    size=len(scope_leagues),
                    environment={_PUBLISH_LEAGUES_ENV_VAR: ",".join(scope_leagues)},
                    depends_on=depends_on,
                )
            )

    return submitted


# Each league-fan-out array child reads its league out of one of these by array
# index, the same trick the optimize array uses for its work list.
_PUBLISH_LEAGUES_ENV_VAR = "CASSANDRA_PUBLISH_LEAGUES"
_ANCHOR_LEAGUES_ENV_VAR = "CASSANDRA_ANCHOR_LEAGUES"


def _league_args(leagues: list[str] | None) -> list[str]:
    """`--league` for the stage that takes its scope on the command line.

    One comma-joined flag rather than one flag per league, because the other
    end is `jobs.py`, and fire keeps only the *last* of a repeated flag:
    `--league nhl --league wnba` arrives as `"wnba"` and the run scores one
    league while reporting success for the set. `--league nhl,wnba` arrives
    as `("nhl", "wnba")`, which is what `_as_list` is written against.
    """
    if not leagues:
        return []
    return ["--league", ",".join(leagues)]


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
            # know which item it is. Pin it to 0 -- under our own name,
            # because Batch reserves the `AWS_BATCH` prefix and drops an
            # override that uses it without saying so. Pinning it under the
            # reserved name is what made a single-league publish quietly
            # republish every league instead.
            request["containerOverrides"]["environment"].append(
                {"name": manifest.PINNED_INDEX_ENV_VAR, "value": "0"}
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
    return _league_at(_PUBLISH_LEAGUES_ENV_VAR, index)


def anchor_league(index: int) -> str:
    """Which league this anchors array child owns."""
    return _league_at(_ANCHOR_LEAGUES_ENV_VAR, index)


def _league_at(env_var: str, index: int) -> str:
    """One league out of a launcher-pinned list, by array index.

    Pinned rather than recomputed for the same reason the optimize manifest
    is: the list the launcher sized the array against is the only list whose
    indices mean what the launcher meant.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        raise ValueError(
            f"{env_var} is not set; this job expects the launcher to pin the "
            "league list it sized the array against."
        )
    leagues = raw.split(",")
    if not 0 <= index < len(leagues):
        raise IndexError(f"Array index {index} is out of range for leagues {leagues}")
    return leagues[index]


def work_for(index: int | None = None) -> Work:
    """Re-exported so the CLI has one place to ask about array indices."""
    return manifest.resolve_index(index)
