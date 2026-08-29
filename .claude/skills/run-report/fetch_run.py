#!/usr/bin/env python3
"""Pull one Batch run out of AWS and cache it on disk for `summarize_run.py`.

A "run" is one `jobs.py submit`: an anchors array, an optimize array, an
evaluate job and a publish array, submitted back-to-back by the launcher. They
carry no shared identifier -- Batch has no notion of a workflow, and
`dag.py:_job_name` stamps each stage from its own `datetime.now()` -- so the
grouping here is by submission time, which is the only thing they share. See
`_group_runs`.

What lands in the cache:

    logs/batch/<run-id>/run.json                      status, timing, names
    logs/batch/<run-id>/optimize-00-mens-elo538.log   one per container
    logs/batch/<run-id>/evaluations.csv               the metrics table, from s3

The csv comes from s3 rather than from a log because that's the only place it
exists: `evaluate_models.py` writes a file and `jobs.py` uploads it, so the
evaluate container's stdout is a list of model names and nothing else.

Only terminal jobs are cached; a RUNNING job's stream is re-fetched every time,
because the interesting question about one is what it has printed since.

    poetry run python fetch_run.py                # newest run, print the path
    poetry run python fetch_run.py --list         # recent runs, newest first
    poetry run python fetch_run.py 20260829-022910

Needs the poetry env (aiobotocore) and AWS credentials: set AWS_PROFILE to an
SSO profile from ~/.aws/config, or pass --profile.
"""

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aiobotocore.session import AioSession, get_session
from botocore.exceptions import BotoCoreError, ClientError

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPO_ROOT / "logs" / "batch"

# Batch's default when a job definition sets no logConfiguration, which none of
# cassandra's do -- see the `batch_job` module's container_properties.
LOG_GROUP = "/aws/batch/job"

# Every state a job can be listed in. `list_jobs` takes exactly one, so
# discovering a run costs one call per state -- they run concurrently.
_STATUSES = (
    "SUBMITTED",
    "PENDING",
    "RUNNABLE",
    "STARTING",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
)

_TERMINAL = {"SUCCEEDED", "FAILED"}

# `cassandra-optimize-20260829-022910` -> stage, stamp. The scheduled launcher
# jobs (`cassandra-optimize-weekly`, `cassandra-publish-daily`) deliberately
# don't match: they're the job that *submits* a run, not part of one.
_JOB_NAME = re.compile(r"^cassandra-(?P<stage>[a-z]+)-(?P<stamp>\d{8}-\d{6})$")

# The stages, in DAG order. Anything else on the queue isn't ours.
STAGES = ("anchors", "optimize", "evaluate", "publish")

# The launcher pins each array's fan-out list into the parent's environment, so
# an index can be turned back into a name without recomputing anything locally.
# Recomputing is exactly the bug `manifest.resolve_index` exists to prevent: a
# summarizer on a different commit than the image would label the rows wrong.
_MANIFEST_VARS = {
    "optimize": ("CASSANDRA_BATCH_MANIFEST", "json"),
    "publish": ("CASSANDRA_PUBLISH_LEAGUES", "csv"),
    "anchors": ("CASSANDRA_ANCHOR_LEAGUES", "csv"),
}

# Two jobs are the same run if they were submitted within this of each other.
# `dag.submit` sends all four in one loop, milliseconds apart; the closest two
# distinct runs observed are minutes apart. Anything in between would be two
# launchers racing, which is worth showing as one run anyway.
_RUN_WINDOW_MS = 60_000

# Enough to keep the round trips overlapping without tripping CloudWatch's
# per-account read throttle, which is what a full 16-child fan-out at
# unbounded concurrency finds. Same shape as `artifacts._MAX_CONCURRENCY`.
_MAX_CONCURRENCY = 16

# Where evaluate's metrics csv lands, mirroring `artifacts.ARTIFACT_PREFIX`
# and the CASSANDRA_HOME-relative path `evaluate_models.py` writes to.
_EVALUATIONS_PREFIX = "cassandra/evaluations/"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class AwsUnavailable(RuntimeError):
    """Credentials or permissions are missing, with the boto message attached."""


@dataclass
class Run:
    """One `jobs.py submit`: the stage jobs that went out together.

    `latest` is scratch for `_group_runs` -- the newest submission folded in so
    far, which is what the next job's gap is measured against.
    """

    run_id: str
    created_at: int
    latest: int
    stages: dict[str, dict]


def _session(profile=None):
    """A session, honouring --profile over AWS_PROFILE over the default."""
    if profile:
        session = AioSession(profile=profile)
        return session
    return get_session()


def default_queue():
    """The queue name, resolved the way `jobs.py:_default_queue` resolves it."""
    from_env = os.environ.get("CASSANDRA_JOB_QUEUE")
    if from_env:
        return from_env

    for candidate in (
        os.environ.get("BATCH_CONFIG_FILE_PATH"),
        str(Path.home() / ".aws-batch" / "config.json"),
    ):
        if candidate and Path(candidate).exists():
            raw = json.loads(Path(candidate).read_text())
            if "job_queue_name" in raw:
                return raw["job_queue_name"]["value"]
    raise SystemExit(
        "No job queue found; set CASSANDRA_JOB_QUEUE or put the terraform "
        "output at ~/.aws-batch/config.json (`make outputs` in "
        "aws-batch-optimization)."
    )


def default_bucket():
    """The batch bucket, from the same config file. None if it isn't there."""
    from_env = os.environ.get("CASSANDRA_BUCKET")
    if from_env:
        return from_env
    for candidate in (
        os.environ.get("BATCH_CONFIG_FILE_PATH"),
        str(Path.home() / ".aws-batch" / "config.json"),
    ):
        if candidate and Path(candidate).is_file():
            raw = json.loads(Path(candidate).read_text())
            if "bucket" in raw:
                return raw["bucket"]["value"]
    return None


async def _list_jobs(batch, queue):
    """Every cassandra job on the queue, across all states."""

    async def one(status):
        found = []
        paginator = batch.get_paginator("list_jobs")
        async for page in paginator.paginate(jobQueue=queue, jobStatus=status):
            found += page.get("jobSummaryList", [])
        return found

    pages = await asyncio.gather(*(one(status) for status in _STATUSES))
    return [
        job for group in pages for job in group if job["jobName"].startswith("cassandra-")
    ]


def _group_runs(jobs):
    """Cluster jobs into runs by submission time.

    Time rather than name, because the four stages stamp their names from
    separate `datetime.now()` calls and land on different seconds often enough
    that grouping on the stamp splits roughly one run in three.
    """
    staged = []
    launchers = []
    for job in jobs:
        match = _JOB_NAME.match(job["jobName"])
        if match and match["stage"] in STAGES:
            staged.append((job, match["stage"], match["stamp"]))
        else:
            # A scheduled launcher. Kept, because a launcher that died is a run
            # that doesn't exist, and that's the answer to "what happened?".
            launchers.append(job)

    staged.sort(key=lambda item: item[0]["createdAt"])
    runs: list[Run] = []
    for job, stage, stamp in staged:
        # `latest` walks forward with each stage so the window is measured from
        # the previous stage, not from the run's first; four stages submitted a
        # second apart are one run however long the loop takes.
        if runs and job["createdAt"] - runs[-1].latest <= _RUN_WINDOW_MS:
            runs[-1].stages[stage] = job
            runs[-1].latest = job["createdAt"]
        else:
            runs.append(
                Run(
                    run_id=stamp,
                    created_at=job["createdAt"],
                    latest=job["createdAt"],
                    stages={stage: job},
                )
            )

    runs.sort(key=lambda run: run.created_at, reverse=True)
    return runs, sorted(launchers, key=lambda job: job["createdAt"], reverse=True)


def _environment(container):
    return {item["name"]: item["value"] for item in container.get("environment", [])}


def _child_names(stage, container, size):
    """Index -> human name for an array stage's children.

    Empty when the launcher pinned nothing, which is what a hand-submitted
    single job looks like; callers fall back to a positional label.
    """
    spec = _MANIFEST_VARS.get(stage)
    if spec is None:
        return {}
    var, encoding = spec
    raw = _environment(container).get(var)
    if raw is None:
        return {}
    names = json.loads(raw) if encoding == "json" else raw.split(",")
    return dict(enumerate(names))


async def _describe(batch, job_ids):
    """describe_jobs, in chunks of 100 (the API's limit), concurrently."""
    chunks = [job_ids[start : start + 100] for start in range(0, len(job_ids), 100)]
    responses = await asyncio.gather(*(batch.describe_jobs(jobs=c) for c in chunks))
    return [job for response in responses for job in response.get("jobs", [])]


async def _array_children(batch, job_id, status_summary):
    """Every child of an array job.

    `list_jobs(arrayJobId=...)` returns nothing at all without a `jobStatus`,
    so the states have to be asked for one at a time -- but `statusSummary` on
    the parent already says which ones hold children, so a finished array costs
    one call rather than seven.
    """
    wanted = [s for s in _STATUSES if status_summary.get(s)] or list(_STATUSES)

    async def one(status):
        found = []
        paginator = batch.get_paginator("list_jobs")
        async for page in paginator.paginate(arrayJobId=job_id, jobStatus=status):
            found += page.get("jobSummaryList", [])
        return found

    pages = await asyncio.gather(*(one(status) for status in wanted))
    return [job for group in pages for job in group]


async def _fetch_stream(logs, stream):
    """Every event in one log stream, oldest first, ANSI stripped.

    bayes_opt colors its probe tables and the codes survive into CloudWatch;
    stripping on write keeps the cached file greppable and roughly halves it.
    """
    lines = []
    token = None
    while True:
        kwargs = {
            "logGroupName": LOG_GROUP,
            "logStreamName": stream,
            "startFromHead": True,
        }
        if token:
            kwargs["nextToken"] = token
        response = await logs.get_log_events(**kwargs)
        lines += [event["message"] for event in response.get("events", [])]
        following = response.get("nextForwardToken")
        # get_log_events hands back the same forward token once it has reached
        # the end, so an unchanged token is the terminator -- an empty page
        # alone isn't, mid-stream.
        if not following or following == token:
            break
        token = following
    return [_ANSI.sub("", line) for line in lines]


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "child"


def _log_name(stage, index, name):
    if index is None:
        return f"{stage}.log"
    return f"{stage}-{index:02d}-{_slug(name)}.log"


async def _job_record(logs, stage, index, name, job, cache_dir, gate):
    """One container's status, timing and log, cached if it is finished."""
    container = job.get("container") or {}
    attempts = job.get("attempts") or []
    record = {
        "index": index,
        "name": name,
        "job_id": job["jobId"],
        "status": job["status"],
        "started_at": job.get("startedAt"),
        "stopped_at": job.get("stoppedAt"),
        "exit_code": container.get("exitCode"),
        # Why Batch says it ended. "Dependent Job failed" and "Cancelled via
        # console" both arrive here with no log at all, and the run report has
        # nothing else to explain the gap with.
        "status_reason": job.get("statusReason"),
        # Why the *container* failed, which is where a pull error lands:
        # `CannotPullImageManifestError` never reaches stdout.
        "container_reason": container.get("reason"),
        # More than one means Batch retried, which it only does on `Host EC2*`
        # -- a spot reclaim. Nothing checkpoints, so the earlier attempt's work
        # is gone rather than resumed.
        "attempts": len(attempts),
        "log": None,
    }

    stream = container.get("logStreamName")
    if not stream:
        return record

    log_name = _log_name(stage, index, name)
    path = cache_dir / log_name
    if job["status"] in _TERMINAL and path.exists():
        record["log"] = log_name
        return record

    async with gate:
        lines = await _fetch_stream(logs, stream)
    if lines or job["status"] in _TERMINAL:
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        record["log"] = log_name
    return record


async def _fetch_stage(batch, logs, stage, parent, cache_dir, gate):
    """One stage: its children's records, in array-index order."""
    array = parent.get("arrayProperties") or {}
    size = array.get("size", 0)
    container = parent.get("container") or {}
    names = _child_names(stage, container, size)

    if size:
        children = await _array_children(batch, parent["jobId"], array.get("statusSummary") or {})
        described = await _describe(batch, [child["jobId"] for child in children])
        described.sort(key=_index_of)
        records = await asyncio.gather(
            *(
                _job_record(
                    logs,
                    stage,
                    _index_of(job),
                    names.get(_index_of(job), f"{stage}[{_index_of(job)}]"),
                    job,
                    cache_dir,
                    gate,
                )
                for job in described
            )
        )
    else:
        records = [
            await _job_record(logs, stage, None, stage, parent, cache_dir, gate)
        ]

    return stage, {
        "job_id": parent["jobId"],
        "job_name": parent["jobName"],
        "status": parent["status"],
        "created_at": parent.get("createdAt"),
        "size": size,
        "command": container.get("command", []),
        "status_summary": array.get("statusSummary") or {},
        "children": list(records),
    }


def _index_of(job):
    return (job.get("arrayProperties") or {}).get("index", 0)


async def _fetch_evaluations(s3, bucket, cache_dir, window):
    """The metrics csv this run produced, if evaluate got that far.

    `window` is the evaluate container's own (started, stopped), because that
    is the only thing that identifies the csv as *this* run's. Neither
    alternative works: the object name is a timestamp the container chose, and
    "the newest csv in the bucket" belongs to whichever run finished last --
    which, for any run that failed, is a later one whose numbers have nothing
    to do with it.
    """
    if bucket is None or window is None:
        return None
    started, stopped = window
    if started is None:
        return None  # evaluate never ran, so it wrote nothing.
    # A generous tail: the upload follows the write, and both are inside the
    # container's life, but clock skew between S3 and Batch is not zero.
    latest = (stopped or started) + 300_000

    paginator = s3.get_paginator("list_objects_v2")
    newest = None
    async for page in paginator.paginate(Bucket=bucket, Prefix=_EVALUATIONS_PREFIX):
        for item in page.get("Contents", []):
            written = item["LastModified"].timestamp() * 1000
            if not started <= written <= latest:
                continue
            if newest is None or item["LastModified"] > newest["LastModified"]:
                newest = item
    if newest is None:
        return None

    written = newest["LastModified"].timestamp() * 1000
    response = await s3.get_object(Bucket=bucket, Key=newest["Key"])
    body = await response["Body"].read()
    (cache_dir / "evaluations.csv").write_bytes(body)
    return {
        "key": newest["Key"],
        "written_at": int(written),
        "file": "evaluations.csv",
    }


async def _fetch(run, profile, cache_root, bucket):
    cache_dir = cache_root / run.run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(_MAX_CONCURRENCY)
    session = _session(profile)

    async with AsyncExitStack() as stack:
        batch = await stack.enter_async_context(session.create_client("batch"))
        logs = await stack.enter_async_context(session.create_client("logs"))
        s3 = await stack.enter_async_context(session.create_client("s3"))

        parents = await _describe(
            batch, [job["jobId"] for job in run.stages.values()]
        )
        by_id = {job["jobId"]: job for job in parents}

        ordered = [stage for stage in STAGES if stage in run.stages]
        results = await asyncio.gather(
            *(
                _fetch_stage(
                    batch,
                    logs,
                    stage,
                    by_id[run.stages[stage]["jobId"]],
                    cache_dir,
                    gate,
                )
                for stage in ordered
            )
        )
        stages = dict(results)

        # The evaluate container's own lifetime, which is the window its csv
        # must fall in. No evaluate stage means no csv belongs to this run.
        window = None
        if "evaluate" in stages:
            children = stages["evaluate"]["children"]
            starts = [c["started_at"] for c in children if c["started_at"]]
            stops = [c["stopped_at"] for c in children if c["stopped_at"]]
            window = (min(starts) if starts else None, max(stops) if stops else None)
        evaluations = await _fetch_evaluations(s3, bucket, cache_dir, window)

    # The exact list the optimize array was sized against, kept so the report
    # can say what a run was *meant* to cover even when most children never
    # started and have no logs to name themselves with.
    manifest = None
    if "optimize" in run.stages:
        parent = by_id[run.stages["optimize"]["jobId"]]
        raw = _environment(parent.get("container") or {}).get("CASSANDRA_BATCH_MANIFEST")
        manifest = json.loads(raw) if raw else None

    payload = {
        "run_id": run.run_id,
        "created_at": run.created_at,
        "manifest": manifest,
        "evaluations": evaluations,
        "stages": stages,
    }
    (cache_dir / "run.json").write_text(json.dumps(payload, indent=2) + "\n")
    return cache_dir, payload


async def _discover(queue, profile):
    session = _session(profile)
    async with session.create_client("batch") as batch:
        return _group_runs(await _list_jobs(batch, queue))


def discover(queue=None, profile=None):
    """Runs on the queue, newest first, plus any launcher jobs."""
    return _wrap(_discover(queue or default_queue(), profile))


def fetch(run, profile=None, cache_root=CACHE_ROOT, bucket=None):
    """Materialize one run's status, logs and metrics under `logs/batch/`."""
    return _wrap(
        _fetch(run, profile, cache_root, bucket if bucket else default_bucket())
    )


def _wrap(coroutine):
    """Run a coroutine, turning credential failures into one clear message."""
    try:
        return asyncio.run(coroutine)
    except (ClientError, BotoCoreError) as error:
        raise AwsUnavailable(str(error)) from error


def load(run_id, cache_root=CACHE_ROOT):
    """A previously fetched run, straight off disk."""
    path = cache_root / run_id / "run.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _stamp(millis):
    if millis is None:
        return "?"
    return datetime.fromtimestamp(millis / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _summary_line(run):
    present = ", ".join(stage for stage in STAGES if stage in run.stages)
    return f"{run.run_id}  {_stamp(run.created_at)}  {present}"


def main(argv):
    profile = os.environ.get("AWS_PROFILE")
    if "--profile" in argv:
        profile = argv[argv.index("--profile") + 1]
    positional = [
        arg
        for index, arg in enumerate(argv[1:], start=1)
        if not arg.startswith("-") and argv[index - 1] != "--profile"
    ]

    try:
        runs, launchers = discover(profile=profile)
    except AwsUnavailable as error:
        raise SystemExit(
            f"{error}\n\nSet AWS_PROFILE to an SSO profile from ~/.aws/config "
            "(or pass --profile), and re-run `aws sso login` if the session "
            "has expired."
        )

    if "--list" in argv:
        for run in runs:
            print(_summary_line(run))
        if not runs:
            print("No cassandra runs on the queue.")
        if launchers:
            print("\nLauncher jobs (these submit runs; they are not one):")
            for job in launchers:
                print(
                    f"  {job['jobName']}  {_stamp(job['createdAt'])}  {job['status']}"
                )
        return

    if not runs:
        raise SystemExit(
            "No cassandra runs on the queue. `make submit` starts one; "
            "--list also shows launcher jobs, which is where to look if a "
            "scheduled run never appeared."
        )

    if positional:
        wanted = positional[0]
        matches = [run for run in runs if run.run_id == wanted]
        if not matches:
            known = ", ".join(run.run_id for run in runs[:10])
            raise SystemExit(f"No run {wanted}. Recent: {known}")
        run = matches[0]
    else:
        run = runs[0]

    cache_dir, _ = fetch(run, profile)
    print(cache_dir)


if __name__ == "__main__":
    main(sys.argv)
