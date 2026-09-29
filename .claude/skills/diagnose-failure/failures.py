#!/usr/bin/env python3
"""Failed Batch jobs, and what one of them printed before it died. Read-only.

    python failures.py                      # failed jobs in the last 24h, root causes first
    python failures.py --since 72h
    python failures.py 20260915-044450      # one run's failures
    python failures.py --ids                # just the root-cause job ids, one per line
    python failures.py <job-id>             # one job: every attempt, condensed log tail
    python failures.py <job-id>:<index>     # one array child
    python failures.py cassandra-launcher   # the newest job by that name
    python failures.py <job-id> --lines 80  # a longer tail

Two modes, both built to put as little as possible in front of whoever reads
the output, because the output is read by a model paying per token:

- **List** reads Batch's job records and nothing else -- no log is fetched.
  It groups failures by run, drops the `Dependent Job failed` casualties to a
  count, and names the job to look at next.
- **Detail** fetches one job's streams, saves each attempt whole to
  `logs/batch/jobs/<job-id>-attempt<N>.log` (for Grep, not for reading
  through), and prints a condensed tail: bayes_opt probe rows, repeated lines
  and repeated warnings collapse to a count, and a traceback keeps the frames
  in cassandra and its own dependencies and folds the library frames between
  them.

Permissions it needs, and all it needs: `batch:ListJobs`,
`batch:DescribeJobs` and `logs:GetLogEvents` on `/aws/batch/job`. The
`cassandra-ci-diagnose` role in `jobs/oidc.tf` grants exactly those. The
queue comes from `CASSANDRA_JOB_QUEUE` or `~/.aws-batch/config.json`.
"""

import asyncio
import re
import sys
import time
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
_RUN_ID = re.compile(r"^\d{8}-\d{6}$")
_SINCE = re.compile(r"^(\d+)([hd])$")

_CASCADE = "Dependent Job failed"

# Which reason names which problem. Checked in order against an attempt's
# container reason, then its status reason; the first hit labels it. The
# SKILL.md table is the long form of each.
_KINDS = (
    ("CannotPullImage", "image not in ECR"),
    ("CannotStartContainer", "container never started"),
    ("ResourceInitializationError", "container never started"),
    ("OutOfMemory", "out of memory"),
    ("Host EC2", "spot reclaim"),
    ("attempt duration exceeded", "timeout"),
    (_CASCADE, "cascade"),
    ("Cancelled", "cancelled"),
    ("Terminated", "terminated"),
)

# Log lines that carry no information one at a time. bayes_opt's table: its
# rows, its header and its borders.
_PROBE_ROW = re.compile(r"^\s*(\|.*\||[-=+]{10,})\s*$")
_WARNING = re.compile(r"^\S+:\d+: \w*Warning: ")
_TRACEBACK = re.compile(r"^Traceback \(most recent call last\)")
_FRAME = re.compile(r'^\s+File "(?P<file>[^"]+)", line \d+')
# Installed packages whose frames are worth keeping in a traceback: the ones
# cassandra pins and would patch. The repo's own files aren't in
# site-packages at all; every other installed frame is folded.
_OWN_PACKAGES = ("/endgame", "/call_it_what_you_want/", "/say_youll_remember_me/")


def _kind(reasons, exit_code=None):
    joined = " ".join(r for r in reasons if r).lower()
    for needle, label in _KINDS:
        if needle.lower() in joined:
            return label
    if exit_code == 137:
        return "killed (137) -- OOM or SIGKILL"
    if exit_code:
        return f"exit {exit_code}"
    return joined or "no reason given"


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


# ------------------------------------------------------------------------------
# Condensing a log
# ------------------------------------------------------------------------------


def _fold_frames(block):
    """A traceback with the library frames between the interesting ones folded.

    A frame is its `File` line plus the source lines under it. Kept: the
    repo's and its own dependencies' frames, and the last frame of all, which
    is where the exception was raised whoever owns it.
    """
    head, frames, tail = [], [], []
    for line in block:
        if _FRAME.match(line):
            frames.append([line])
        elif frames and line.startswith("    ") and not tail:
            frames[-1].append(line)
        elif frames:
            tail.append(line)
        else:
            head.append(line)

    out = list(head)
    folded = 0
    for index, frame in enumerate(frames):
        path = _FRAME.match(frame[0])["file"]
        own = "site-packages" not in path or any(p in path for p in _OWN_PACKAGES)
        if own or index == len(frames) - 1:
            if folded:
                out.append(f"    [... {folded} library frames ...]")
                folded = 0
            out += frame
        else:
            folded += 1
    return out + tail


def condense(lines):
    """The lines a reader needs, with the noise counted rather than shown."""
    out = []
    probes = 0
    seen_warnings = {}
    last = None
    repeats = 0

    def flush():
        nonlocal probes, repeats
        if repeats:
            out.append(f"[... previous line x{repeats} more ...]")
            repeats = 0
        if probes:
            out.append(f"[... {probes} probe-table rows ...]")
            probes = 0

    for line in lines:
        if _PROBE_ROW.match(line):
            probes += 1
            continue
        if line == last:
            repeats += 1
            continue
        if _WARNING.match(line):
            key = line
            seen_warnings[key] = seen_warnings.get(key, 0) + 1
            if seen_warnings[key] > 1:
                continue
        flush()
        out.append(line)
        last = line
    flush()

    dropped = sum(n - 1 for n in seen_warnings.values())
    if dropped:
        out.append(f"[{dropped} repeats of warnings shown above were dropped]")

    # Chained tracebacks put the exception that escaped last.
    starts = [i for i, line in enumerate(out) if _TRACEBACK.match(line)]
    if starts:
        out = out[: starts[-1]] + _fold_frames(out[starts[-1] :])
    return out, bool(starts)


def _tail(condensed, traced, limit):
    """From the last traceback on, or the last `limit` lines."""
    if traced:
        start = max(i for i, line in enumerate(condensed) if _TRACEBACK.match(line))
        # A little context before it: what the job was doing when it broke.
        return condensed[max(0, start - 5) :][-limit:]
    return condensed[-limit:]


# ------------------------------------------------------------------------------
# AWS
# ------------------------------------------------------------------------------


async def _stream(logs, stream):
    """(timestamp, message) for every event, oldest first, ANSI stripped."""
    events = []
    token = None
    while True:
        kwargs = {
            "logGroupName": fetch_run.LOG_GROUP,
            "logStreamName": stream,
            "startFromHead": True,
        }
        if token:
            kwargs["nextToken"] = token
        response = await logs.get_log_events(**kwargs)
        events += [
            (e["timestamp"], fetch_run._ANSI.sub("", e["message"]))
            for e in response.get("events", [])
        ]
        following = response.get("nextForwardToken")
        if not following or following == token:
            break
        token = following
    return events


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


def _attempts(job):
    """One row per attempt; one synthetic row for a job that never started."""
    rows = []
    for number, attempt in enumerate(job.get("attempts") or [], start=1):
        container = attempt.get("container") or {}
        rows.append(
            {
                "number": number,
                "exit": container.get("exitCode"),
                "reason": container.get("reason"),
                "status_reason": attempt.get("statusReason"),
                "stream": container.get("logStreamName"),
                "started": attempt.get("startedAt"),
                "stopped": attempt.get("stoppedAt"),
            }
        )
    if not rows:
        container = job.get("container") or {}
        rows.append(
            {
                "number": 0,
                "exit": container.get("exitCode"),
                "reason": container.get("reason"),
                "status_reason": job.get("statusReason"),
                "stream": container.get("logStreamName"),
                "started": None,
                "stopped": None,
            }
        )
    return rows


def _job_kind(job):
    last = _attempts(job)[-1]
    return _kind([last["reason"], last["status_reason"]], last["exit"])


def _stage(name):
    match = fetch_run._JOB_NAME.match(name)
    return match["stage"] if match else None


# ------------------------------------------------------------------------------
# List mode
# ------------------------------------------------------------------------------


async def _list(batch, queue, run_id, since_ms, ids_only):
    paginator = batch.get_paginator("list_jobs")
    summaries = []
    async for page in paginator.paginate(jobQueue=queue, jobStatus="FAILED"):
        summaries += [
            j
            for j in page.get("jobSummaryList", [])
            if j["jobName"].startswith("cassandra-")
        ]
    jobs = await fetch_run._describe(batch, [j["jobId"] for j in summaries])
    runs, launchers = fetch_run._group_runs(jobs)

    if run_id:
        runs = [r for r in runs if r.run_id == run_id]
        launchers = []
    else:
        runs = [r for r in runs if r.created_at >= since_ms]
        launchers = [j for j in launchers if j["createdAt"] >= since_ms]

    lines, roots = [], []
    for job in launchers:
        roots.append(job["jobId"])
        lines += [
            f"LAUNCHER {job['jobName']}  {_stamp(job['createdAt'])}  "
            f"{_job_kind(job)}  -- no run was submitted",
            f"  next: failures.py {job['jobId']}",
        ]

    for run in runs:
        order = [s for s in fetch_run.STAGES if s in run.stages]
        rows, root, reclaimed = [], None, None
        for stage in order:
            parent = run.stages[stage]
            size = (parent.get("arrayProperties") or {}).get("size")
            children = await _failed_children(batch, parent) if size else [parent]
            real = [c for c in children if _job_kind(c) != "cascade"]
            cascaded = len(children) - len(real)
            if not real:
                rows.append(f"  {stage:<12} {cascaded} cascaded")
                continue
            names = fetch_run._child_names(
                stage, parent.get("container") or {}, size or 0
            )
            groups = {}
            for child in real:
                groups.setdefault(_job_kind(child), []).append(child)
            for kind, members in groups.items():
                labels = [
                    names.get(fetch_run._index_of(c), f"[{fetch_run._index_of(c)}]")
                    if size
                    else stage
                    for c in members
                ]
                shown = ", ".join(labels[:6]) + (
                    f" +{len(labels) - 6}" if len(labels) > 6 else ""
                )
                rows.append(f"  {stage:<12} {len(members)} {kind}: {shown}")
                if kind == "spot reclaim":
                    reclaimed = reclaimed or members[0]["jobId"]
                elif root is None:
                    root = members[0]["jobId"]
            if cascaded:
                rows.append(f"  {stage:<12} {cascaded} cascaded")
        # Only reclaims failed: still the thing to look at.
        root = root or reclaimed
        if root:
            roots.append(root)
        lines += [f"RUN {run.run_id}  {_stamp(run.created_at)}", *rows]
        if root:
            lines.append(f"  next: failures.py {root}")

    if ids_only:
        print("\n".join(roots))
        return
    if not lines:
        scope = f"run {run_id}" if run_id else f"since {_stamp(since_ms)}"
        print(f"No failed cassandra jobs ({scope}).")
        return
    print("\n".join(lines))


# ------------------------------------------------------------------------------
# Detail mode
# ------------------------------------------------------------------------------


async def _detail(logs, job, limit):
    container = job.get("container") or {}
    memory = container.get("memory") or next(
        (
            r["value"]
            for r in container.get("resourceRequirements") or []
            if r.get("type") == "MEMORY"
        ),
        "?",
    )
    timeout = (job.get("timeout") or {}).get("attemptDurationSeconds")
    out = [
        f"JOB {job['jobName']}  {job['jobId']}  {job['status']}",
        f"  definition {job.get('jobDefinition', '?').rsplit('/', 1)[-1]}"
        f"  image {container.get('image', '?').rsplit('/', 1)[-1]}"
        f"  command {' '.join(container.get('command') or [])}",
        f"  memory {memory} MiB" + (f"  timeout {timeout}s" if timeout else ""),
    ]

    rows = _attempts(job)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = job["jobId"].replace(":", "-")
    for row in rows:
        reason = " / ".join(r for r in (row["status_reason"], row["reason"]) if r)
        out.append(
            f"#{row['number']} exit={row['exit']} "
            f"{_span(row['started'], row['stopped'])} "
            f"[{_kind([row['reason'], row['status_reason']], row['exit'])}] {reason}"
        )
        if not row["stream"]:
            out.append("  no log stream")
            continue
        events = await _stream(logs, row["stream"])
        lines = [message for _, message in events]
        path = OUT_DIR / f"{slug}-attempt{row['number']}.log"
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        # Relative, so it's what a Grep or Read call takes from the repo root.
        if path.is_relative_to(fetch_run.REPO_ROOT):
            path = path.relative_to(fetch_run.REPO_ROOT)
        if events and row["stopped"]:
            gap = _span(events[-1][0], row["stopped"])
            out.append(
                f"  log: {len(lines)} lines, last output {gap} before stop, {path}"
            )
        else:
            out.append(f"  log: {len(lines)} lines, {path}")
        if row is not rows[-1]:
            # Earlier attempts: their last line tells a reclaim from a crash.
            if lines:
                out.append(f"  last: {lines[-1][:200]}")
            continue
        condensed, traced = condense(lines)
        tail = _tail(condensed, traced, limit)
        out += [f"  | {line[:300]}" for line in tail] or ["  | (empty)"]
    return out


async def _detail_mode(batch, logs, target, limit):
    job_id = (
        target
        if _JOB_ID.match(target)
        else await _find(batch, fetch_run.default_queue(), target)
    )
    described = await fetch_run._describe(batch, [job_id])
    if not described:
        raise SystemExit(f"Batch has no job {job_id} (it keeps them about 7 days).")
    job = described[0]

    size = (job.get("arrayProperties") or {}).get("size")
    if not size:
        print("\n".join(await _detail(logs, job, limit)))
        return

    # An array parent has no log. Group its failed children and show the
    # first of each group that isn't a casualty.
    failed = await _failed_children(batch, job)
    names = fetch_run._child_names(
        _stage(job["jobName"]) or "", job.get("container") or {}, size
    )
    print(f"ARRAY {job['jobName']}  {job['jobId']}  {len(failed)} of {size} failed")
    groups = {}
    for child in failed:
        groups.setdefault(_job_kind(child), []).append(child)
    for kind, members in groups.items():
        labels = [
            names.get(fetch_run._index_of(c), f"[{fetch_run._index_of(c)}]")
            for c in members
        ]
        print(f"  {len(members)} {kind}: {', '.join(labels)}")
    for kind, members in groups.items():
        if kind == "cascade":
            continue
        print()
        print("\n".join(await _detail(logs, members[0], limit)))


# ------------------------------------------------------------------------------


async def _main(argv):
    limit = 40
    since_ms = int((time.time() - 24 * 3600) * 1000)
    profile = None
    positional = []
    args = iter(argv)
    for arg in args:
        if arg == "--lines":
            limit = int(next(args))
        elif arg == "--profile":
            profile = next(args)
        elif arg == "--since":
            match = _SINCE.match(next(args, ""))
            if not match:
                raise SystemExit("--since takes e.g. 24h or 3d")
            hours = int(match[1]) * (24 if match[2] == "d" else 1)
            since_ms = int((time.time() - hours * 3600) * 1000)
        elif arg in ("-h", "--help"):
            raise SystemExit(__doc__)
        elif not arg.startswith("-"):
            positional.append(arg)
    target = positional[0] if positional else None

    session = fetch_run._session(profile)
    async with AsyncExitStack() as stack:
        batch = await stack.enter_async_context(session.create_client("batch"))
        if target is None or _RUN_ID.match(target):
            await _list(
                batch, fetch_run.default_queue(), target, since_ms, "--ids" in argv
            )
            return
        logs = await stack.enter_async_context(session.create_client("logs"))
        await _detail_mode(batch, logs, target, limit)


def main(argv):
    try:
        asyncio.run(_main(argv[1:]))
    except (fetch_run.ClientError, fetch_run.BotoCoreError) as error:
        raise SystemExit(
            f"{error}\n\nSet AWS_PROFILE to an SSO profile from ~/.aws/config "
            "(or pass --profile) and re-run `aws sso login` if it has expired; "
            "in CI, assume the cassandra-ci-diagnose role."
        )


if __name__ == "__main__":
    main(sys.argv)
