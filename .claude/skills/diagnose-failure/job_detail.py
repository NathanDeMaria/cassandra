#!/usr/bin/env python3
"""Everything Batch knows about one failed job, plus where its log broke.

    poetry run python job_detail.py <job-id>            # from the failure email
    poetry run python job_detail.py <job-id>:<index>    # one array child
    poetry run python job_detail.py cassandra-launcher  # newest job by that name
    poetry run python job_detail.py <job-id> --lines 200

`make report` answers "what happened to this run", and it only knows about
runs: the jobs `dag.submit` stamped `cassandra-<stage>-<stamp>`. Two things a
failure question needs fall outside that:

- **The launcher.** `cassandra-launcher` is what the schedules start, and a
  launcher that dies submits nothing -- so there is no run for the report to
  find, and its log is nowhere in `logs/batch/`.
- **Attempts.** The report keeps one status per child. A job that was
  reclaimed four times and then hit a traceback has five reasons, and the
  last one is not always the interesting one.

So this takes one job -- by id, which is what the SNS email carries, or by
name -- and prints its definition, image and command, every attempt's exit
code and reasons, and the tail of each attempt's stream from the last
traceback on. The whole stream is written to `logs/batch/jobs/` so a
follow-up can Read it rather than re-fetch it.

An array parent has no log of its own; for one, this lists its failed
children grouped by reason and details the first of each group.
"""

import asyncio
import re
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run-report"))

import fetch_run  # noqa: E402

OUT_DIR = fetch_run.CACHE_ROOT / "jobs"

# `<uuid>` or `<uuid>:<index>`, the forms Batch hands out.
_JOB_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(:\d+)?$"
)

_TRACEBACK = re.compile(r"^Traceback \(most recent call last\)")

# Which reason names which problem. Checked in order against the attempt's
# container reason, then its status reason; the first hit labels it. The
# SKILL.md table is the long form of each.
_KINDS = (
    ("CannotPullImage", "image not in ECR -- a deploy problem, nothing ran"),
    ("CannotStartContainer", "container never started -- job definition or image"),
    (
        "ResourceInitializationError",
        "container never started -- secrets, network or IAM",
    ),
    (
        "OutOfMemory",
        "out of memory -- the stage's memory variable in jobs/variables.tf",
    ),
    ("Host EC2", "spot reclaim -- retried automatically"),
    ("attempt duration exceeded", "hit the job definition's timeout"),
    ("Dependent Job failed", "cascade -- something upstream failed, look there"),
    ("Cancelled", "cancelled by hand"),
    ("Terminated", "terminated by hand"),
)


def _kind(reasons):
    joined = " ".join(r for r in reasons if r)
    for needle, label in _KINDS:
        if needle.lower() in joined.lower():
            return label
    return None


def _stamp(millis):
    if not millis:
        return "?"
    return datetime.fromtimestamp(millis / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _span(start, stop):
    if not start or not stop:
        return "?"
    seconds = int((stop - start) / 1000)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


def _tail(lines, limit):
    """From the last traceback to the end, or just the end if there is none.

    Chained tracebacks put the exception that escaped last, so the last
    `Traceback` line is where the useful part starts. A traceback longer than
    the limit keeps its end, which is where the exception and the innermost
    frames are.
    """
    starts = [i for i, line in enumerate(lines) if _TRACEBACK.match(line)]
    chunk = lines[starts[-1] :] if starts else lines
    return chunk[-limit:], bool(starts)


async def _find(batch, queue, name):
    """The newest job on the queue with this exact name."""
    found = []
    paginator = batch.get_paginator("list_jobs")
    async for page in paginator.paginate(
        jobQueue=queue, filters=[{"name": "JOB_NAME", "values": [name]}]
    ):
        found += page.get("jobSummaryList", [])
    if not found:
        raise SystemExit(f"No job named {name} on {queue}.")
    return max(found, key=lambda job: job["createdAt"])["jobId"]


async def _failed_children(batch, job):
    children = await fetch_run._array_children(batch, job["jobId"], {"FAILED": 1})
    described = await fetch_run._describe(batch, [c["jobId"] for c in children])
    return sorted(described, key=fetch_run._index_of)


def _attempt_rows(job):
    attempts = job.get("attempts") or []
    rows = []
    for number, attempt in enumerate(attempts, start=1):
        container = attempt.get("container") or {}
        rows.append(
            {
                "number": number,
                "exit": container.get("exitCode"),
                "reason": container.get("reason"),
                "status_reason": attempt.get("statusReason"),
                "stream": container.get("logStreamName"),
                "span": _span(attempt.get("startedAt"), attempt.get("stoppedAt")),
            }
        )
    if not rows:
        # Never placed: no attempt, and the only reasons are on the job.
        container = job.get("container") or {}
        rows.append(
            {
                "number": 0,
                "exit": container.get("exitCode"),
                "reason": container.get("reason"),
                "status_reason": job.get("statusReason"),
                "stream": container.get("logStreamName"),
                "span": "never started",
            }
        )
    return rows


async def _detail(logs, job, limit):
    container = job.get("container") or {}
    env = fetch_run._environment(container)
    out = [
        f"JOB {job['jobName']}  {job['jobId']}",
        f"  status      {job['status']}  ({job.get('statusReason') or 'no reason given'})",
        f"  created     {_stamp(job.get('createdAt'))}",
        f"  ran         {_stamp(job.get('startedAt'))} -> {_stamp(job.get('stoppedAt'))}"
        f"  ({_span(job.get('startedAt'), job.get('stoppedAt'))})",
        f"  definition  {job.get('jobDefinition', '?').rsplit('/', 1)[-1]}",
        f"  image       {container.get('image', '?')}",
        f"  command     {' '.join(container.get('command') or [])}",
        f"  memory      {container.get('memory') or _requirement(container, 'MEMORY')} MiB",
    ]
    timeout = (job.get("timeout") or {}).get("attemptDurationSeconds")
    if timeout:
        out.append(f"  timeout     {timeout}s per attempt")
    index = env.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if index is not None:
        out.append(f"  array index {index}")
    depends = [d["jobId"] for d in job.get("dependsOn") or []]
    if depends:
        out.append(f"  depends on  {', '.join(depends)}")

    rows = _attempt_rows(job)
    out += ["", f"ATTEMPTS ({len([r for r in rows if r['number']])})"]
    for row in rows:
        kind = _kind([row["reason"], row["status_reason"]])
        out.append(
            f"  #{row['number']}  exit={row['exit']}  {row['span']}  "
            f"{row['status_reason'] or ''}"
            + (f"  [{row['reason']}]" if row["reason"] else "")
            + (f"\n       -> {kind}" if kind else "")
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = job["jobId"].replace(":", "-")
    for row in rows:
        if not row["stream"]:
            continue
        lines = await fetch_run._fetch_stream(logs, row["stream"])
        path = OUT_DIR / f"{slug}-attempt{row['number']}.log"
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        # Only the last attempt's tail is printed in full; earlier ones get
        # their final line, which is enough to tell a reclaim from a crash.
        if row is not rows[-1]:
            last = lines[-1] if lines else "(empty)"
            out += ["", f"LOG #{row['number']} ({len(lines)} lines, {path}): {last}"]
            continue
        tail, traced = _tail(lines, limit)
        what = "from the last traceback" if traced else f"last {len(tail)} lines"
        out += ["", f"LOG #{row['number']} ({len(lines)} lines, {what}; full: {path})"]
        out += [f"  {line}" for line in tail] or [
            "  (empty -- the container wrote nothing)"
        ]
    if not any(row["stream"] for row in rows):
        out += ["", "LOG: none -- the container never produced a stream"]
    return out


def _requirement(container, kind):
    for item in container.get("resourceRequirements") or []:
        if item.get("type") == kind:
            return item.get("value")
    return "?"


async def _main(target, limit, profile):
    session = fetch_run._session(profile)
    async with AsyncExitStack() as stack:
        batch = await stack.enter_async_context(session.create_client("batch"))
        logs = await stack.enter_async_context(session.create_client("logs"))

        job_id = (
            target
            if _JOB_ID.match(target)
            else await _find(batch, fetch_run.default_queue(), target)
        )
        described = await fetch_run._describe(batch, [job_id])
        if not described:
            raise SystemExit(
                f"Batch has no job {job_id} (it forgets them after ~7 days)."
            )
        job = described[0]

        if not (job.get("arrayProperties") or {}).get("size"):
            print("\n".join(await _detail(logs, job, limit)))
            return

        summary = job["arrayProperties"].get("statusSummary") or {}
        names = fetch_run._child_names(
            _stage(job["jobName"]),
            job.get("container") or {},
            job["arrayProperties"]["size"],
        )
        print(
            f"ARRAY {job['jobName']}  {job['jobId']}  {job['status']}  "
            + ", ".join(f"{k.lower()} {v}" for k, v in summary.items() if v)
        )
        failed = await _failed_children(batch, job) if summary.get("FAILED") else []
        if not failed:
            print("  no failed children")
            return
        groups = {}
        for child in failed:
            rows = _attempt_rows(child)
            # Exit code in the key: a crash says nothing in its reasons but
            # "Essential container in task exited", and exit 1 (a traceback)
            # is a different problem from exit 137 with no OOM reason.
            key = _kind([rows[-1]["reason"], rows[-1]["status_reason"]]) or (
                f"exit {rows[-1]['exit']}: "
                f"{rows[-1]['status_reason'] or 'no reason given'}"
            )
            groups.setdefault(key, []).append(child)
        for key, children in groups.items():
            labels = [
                f"[{fetch_run._index_of(c)}] {names.get(fetch_run._index_of(c), '')}".strip()
                for c in children
            ]
            print(f"\n{len(children)} failed: {key}")
            print("  " + ", ".join(labels))
        for key, children in groups.items():
            print("\n" + "=" * 78)
            print("\n".join(await _detail(logs, children[0], limit)))


def _stage(name):
    match = fetch_run._JOB_NAME.match(name)
    return match["stage"] if match else ""


def main(argv):
    profile = None
    limit = 80
    if "--profile" in argv:
        profile = argv[argv.index("--profile") + 1]
    if "--lines" in argv:
        limit = int(argv[argv.index("--lines") + 1])
    positional = [
        arg
        for index, arg in enumerate(argv[1:], start=1)
        if not arg.startswith("-") and argv[index - 1] not in ("--profile", "--lines")
    ]
    if not positional:
        raise SystemExit(__doc__)
    try:
        asyncio.run(_main(positional[0], limit, profile))
    except (fetch_run.ClientError, fetch_run.BotoCoreError) as error:
        raise SystemExit(
            f"{error}\n\nSet AWS_PROFILE to an SSO profile from ~/.aws/config "
            "(or pass --profile), and re-run `aws sso login` if it has expired."
        )


if __name__ == "__main__":
    main(sys.argv)
