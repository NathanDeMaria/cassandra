#!/usr/bin/env python3
"""Condense one AWS Batch run into a report small enough to reason about.

The container logs are mostly bayes_opt probe tables -- a few hundred KB of
numbers whose only interesting rows are the best one and whatever `[optimize]`
printed afterwards. This pulls out the parts a reviewer acts on: per-model
status, convergence, bound diagnostics, grouped failures, and the evaluation
table.

    poetry run python summarize_run.py                # newest run
    poetry run python summarize_run.py 20260829-022910
    poetry run python summarize_run.py --cached       # don't call AWS

Status and timing come from Batch itself (`run.json`), not from scraping the
logs: a job's own `startedAt`/`stoppedAt` and `status` are authoritative, and
the array children run in parallel, so there is no sequential log to infer them
from. `fetch_run.py` puts both on disk.
"""

import csv
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fetch_run

REPO_ROOT = Path(__file__).resolve().parents[3]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# `=== mens/elo538 (Elo538Predictor, n_iter=60) ===`, from `jobs.py`. The
# spaces around the slash are optional because `jobs.py` prints `work.name`
# (no spaces) for a model and `f"{league} / division anchors"` (spaces) for an
# anchor fit, and both come through the same parser.
_SECTION = re.compile(
    r"^=== (?P<league>[^/\s]+) ?/ ?(?P<model>\S+) "
    r"\((?P<cls>\w+), n_iter=(?P<n_iter>\d+)\) ===$"
)
# `| 61        | -0.179261 | 106.28371 | 59.809930 |`
_PROBE = re.compile(r"^\|\s*(\d+)\s*\|\s*(-?[\d.]+(?:e[-+]?\d+)?)\s*\|")
# The last line a finished optimize child prints: `artifacts.upload` echoing
# the key it wrote. Its absence in a SUCCEEDED job means --upload=False.
_UPLOADED = re.compile(r"^\s+uploaded (?P<key>s3://\S+)$")
_DIAGNOSTIC = re.compile(r"^\[optimize\] (?P<message>.+)$")
_WARNING = re.compile(r"^(?P<origin>\S+?:\d+): (?P<kind>\w*Warning): (?P<message>.+)$")
_EXCEPTION = re.compile(
    r"^(?P<type>[\w.]+(?:Error|Exception|Exit|Interrupt)):\s?(?P<message>.*)$"
)
_FRAME = re.compile(r'^\s+File "(?P<file>[^"]+)", line (?P<line>\d+)')
_EVALUATING = re.compile(r"^Evaluating (?P<league>[^:]+): (?P<model>\S+)$")

# Batch states that mean the job is still on its way to running. A whole array
# sitting here is a capacity story, not a model story: the queue is all spot
# `.large`, so RUNNABLE means nothing is being placed.
_WAITING = {"SUBMITTED", "PENDING", "RUNNABLE", "STARTING"}


class Child:
    """One container: its Batch record, plus whatever its log said."""

    def __init__(self, stage, record):
        self.stage = stage
        self.index = record["index"]
        self.name = record["name"]
        self.status = record["status"]
        self.exit_code = record["exit_code"]
        self.status_reason = record["status_reason"]
        self.container_reason = record["container_reason"]
        self.attempts = record["attempts"]
        self.log = record["log"]
        self.started_at = record["started_at"]
        self.stopped_at = record["stopped_at"]

        # Filled in from the log, when there is one.
        self.predictor_class = None
        self.n_iter = None
        self.targets = []
        self.diagnostics = []
        self.error = None
        self.uploaded = None

    @property
    def league(self):
        return self.name.split("/")[0]

    @property
    def duration(self):
        if self.started_at is None:
            return None
        end = self.stopped_at or datetime.now(timezone.utc).timestamp() * 1000
        return timedelta(milliseconds=end - self.started_at)

    @property
    def best(self):
        return max(self.targets) if self.targets else None

    @property
    def best_iteration(self):
        return self.targets.index(self.best) + 1 if self.targets else None

    @property
    def last_improvement(self):
        """The last probe that raised the best-so-far.

        Near the end means the search was still climbing when it ran out of
        iterations; well before it means more probes would have been wasted.
        """
        if not self.targets:
            return None
        best = self.targets[0]
        last = 1
        for index, target in enumerate(self.targets[1:], start=2):
            if target > best:
                best, last = target, index
        return last

    @property
    def gain(self):
        return None if not self.targets else self.best - self.targets[0]

    @property
    def failed(self):
        return self.status == "FAILED"

    @property
    def waiting(self):
        return self.status in _WAITING


def _parse_log(child, lines, warnings):
    """Pull the interesting rows out of one container's stdout."""
    pending_frames = []
    for raw in lines:
        line = _ANSI.sub("", raw.rstrip("\n"))

        section = _SECTION.match(line)
        if section:
            child.predictor_class = section["cls"]
            child.n_iter = int(section["n_iter"])
            pending_frames = []
            continue

        probe = _PROBE.match(line)
        if probe:
            child.targets.append(float(probe.group(2)))
            continue

        uploaded = _UPLOADED.match(line)
        if uploaded:
            child.uploaded = uploaded["key"]
            continue

        diagnostic = _DIAGNOSTIC.match(line)
        if diagnostic:
            child.diagnostics.append(diagnostic["message"])
            continue

        warning = _WARNING.match(line)
        if warning:
            key = (warning["kind"], warning["message"], warning["origin"])
            warnings[key] = warnings.get(key, 0) + 1
            continue

        frame = _FRAME.match(line)
        if frame:
            pending_frames.append(f"{frame['file']}:{frame['line']}")
            if len(pending_frames) > 12:
                pending_frames.pop(0)
            continue

        exception = _EXCEPTION.match(line)
        if exception:
            # Chained tracebacks end with the exception that actually escaped,
            # so later matches win.
            child.error = (
                exception["type"],
                exception["message"],
                tuple(pending_frames[-3:]),
            )
            pending_frames = []
            continue


def _load(cache_dir, payload):
    """Every stage's children, with their logs parsed."""
    warnings = {}
    stages = {}
    evaluated = []
    for stage, record in payload["stages"].items():
        children = []
        for entry in record["children"]:
            child = Child(stage, entry)
            if child.log:
                path = cache_dir / child.log
                if path.exists():
                    lines = path.read_text(errors="replace").splitlines()
                    _parse_log(child, lines, warnings)
                    if stage == "evaluate":
                        evaluated += [
                            f"{m['league']}/{m['model']}"
                            for m in (_EVALUATING.match(line) for line in lines)
                            if m
                        ]
            children.append(child)
        stages[stage] = {"record": record, "children": children}
    return stages, warnings, evaluated


def _duration(delta):
    if delta is None:
        return "?"
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _stamp(millis):
    if millis is None:
        return "?"
    return datetime.fromtimestamp(millis / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _metric(value):
    """Round CSV floats for display, leaving counts and text alone."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return value if number.is_integer() else f"{number:.4f}"


def _optional_float(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    # A metric a league had no data for is written as nan, which formats as
    # "nan" rather than dropping out of the line.
    return None if value != value else value


def _accuracy_by_league(rows):
    """Per-league headline: the best model, and how it did against the market.

    One line per league is what someone asking "how good is the football model"
    actually wants; the full table above is for chasing down why.
    """
    leagues = {}
    for row in rows:
        try:
            brier = float(row["brier_score"])
        except (KeyError, TypeError, ValueError):
            continue
        league = row.get("league", "?")
        if league not in leagues or brier < leagues[league][0]:
            leagues[league] = (brier, row)
    if not leagues:
        return []

    out = ["", "  ACCURACY BY LEAGUE  (best model by brier score)"]
    for league, (brier, row) in sorted(leagues.items()):
        parts = [f"brier {brier:.4f}"]
        accuracy = _optional_float(row, "against_spread_accuracy")
        if accuracy is not None:
            parts.append(f"ATS {accuracy:.1%}")
        # The market only quoted the lined games, so the comparison has to be
        # made on that slice. margin_mae covers every game, so it's the wrong
        # side of the comparison -- except in csvs written before the fitter
        # moved to margin, where margin_mae *was* the lined-games number and
        # spread_game_margin_mae doesn't exist.
        market = _optional_float(row, "market_margin_mae")
        lined_margin = _optional_float(row, "spread_game_margin_mae")
        if lined_margin is None and market is not None:
            lined_margin = _optional_float(row, "margin_mae")
        if lined_margin is not None and market is not None:
            # Positive means the model is further off than the line.
            parts.append(
                f"margin MAE {lined_margin:.2f} vs market {market:.2f}"
                f" ({lined_margin - market:+.2f})"
            )
        else:
            margin = _optional_float(row, "margin_mae")
            if margin is not None:
                parts.append(f"margin MAE {margin:.2f} (no lines to compare against)")
        out.append(
            f"    {league}: {row.get('model', '?')}"
            f" [{row.get('fitter', '?')}] -- " + ", ".join(parts)
        )
    out.append(
        "    (ATS near 50% is noise; a margin MAE at or above the market's means"
    )
    out.append("     the model carries no information the line doesn't already have)")
    return out


def _table(rows, headers):
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths)).rstrip()]
    for row in rows:
        lines.append(
            "  ".join(
                str(cell).ljust(width) for cell, width in zip(row, widths)
            ).rstrip()
        )
    return lines


def _stage_line(stage, entry):
    """One line per DAG node: how its children ended up."""
    children = entry["children"]
    counts = {}
    for child in children:
        counts[child.status] = counts.get(child.status, 0) + 1
    breakdown = ", ".join(f"{count} {status.lower()}" for status, count in counts.items())
    started = [c.started_at for c in children if c.started_at]
    stopped = [c.stopped_at for c in children if c.stopped_at]
    span = ""
    if started and stopped and len(stopped) == len(children):
        # Wall clock for the stage, not the sum of its children: the array runs
        # in parallel, so the sum is what it would have cost sequentially.
        span = f"  wall {_duration(timedelta(milliseconds=max(stopped) - min(started)))}"
    return f"  {stage:<9} {entry['record']['status']:<10} {breakdown}{span}"


def _infrastructure_notes(stages):
    """What went wrong outside the code: reclaims, and containers that never ran.

    Deliberately narrow. A job that failed with a traceback is a FAILURES
    entry, and a whole DAG reading "Dependent Job failed" is one root cause
    grouped there, not twenty-one findings here. What lands here is what no
    traceback can express: an attempt Batch threw away, and a container the
    agent could not start.
    """
    reclaimed = []
    unstartable = {}
    for stage, entry in stages.items():
        for child in entry["children"]:
            label = f"{stage}/{child.name}"
            if child.attempts > 1:
                # `retry_strategy` only retries on `Host EC2*`, and nothing
                # checkpoints, so an earlier attempt's search was discarded --
                # true even when the job eventually SUCCEEDED, which is why
                # this can't be folded into the failure grouping.
                reclaimed.append((label, child.attempts, child.duration))
            if child.container_reason:
                unstartable.setdefault(child.container_reason, []).append(label)

    out = []
    for label, attempts, duration in reclaimed:
        out.append(
            f"  {label}: {attempts} attempts -- spot reclaim. Nothing "
            f"checkpoints, so only the last attempt's {_duration(duration)} "
            "produced anything."
        )
    for reason, labels in unstartable.items():
        shown = ", ".join(labels[:4]) + (
            f", +{len(labels) - 4} more" if len(labels) > 4 else ""
        )
        out.append(f"  container never started x{len(labels)}: {shown}")
        out.append(f"    {reason}")
    return out


def _root_cause(stages):
    """The earliest stage that failed on its own, for a cascaded run.

    Batch marks every downstream job "Dependent Job failed", so the DAG order
    is what says which one was the cause rather than a casualty.
    """
    cascaded = any(
        child.status_reason == "Dependent Job failed"
        for entry in stages.values()
        for child in entry["children"]
    )
    if not cascaded:
        return None
    for stage in fetch_run.STAGES:
        entry = stages.get(stage)
        if entry is None:
            continue
        causes = [
            child
            for child in entry["children"]
            if child.failed and child.status_reason != "Dependent Job failed"
        ]
        if causes:
            reasons = sorted({c.status_reason or "no reason given" for c in causes})
            return (
                f"{stage} failed ({len(causes)} of {len(entry['children'])}: "
                f"{'; '.join(reasons)}) and every later stage was cancelled with "
                "it -- nothing downstream ran, so their failures say nothing"
            )
    return None


def _report(cache_dir, payload, stages, warnings, evaluated, evaluation):
    optimize = stages.get("optimize", {}).get("children", [])
    ok = [c for c in optimize if c.status == "SUCCEEDED"]
    failed = [c for c in optimize if c.failed]
    running = [c for c in optimize if c.status == "RUNNING"]
    waiting = [c for c in optimize if c.waiting]

    out = [f"RUN  {payload['run_id']}   ({cache_dir})"]
    out.append(f"submitted  {_stamp(payload['created_at'])}")

    statuses = {entry["record"]["status"] for entry in stages.values()}
    if statuses <= {"SUCCEEDED"}:
        state = "COMPLETE"
    elif statuses & {"RUNNING", "STARTING"}:
        state = "IN PROGRESS"
    elif statuses & {"SUBMITTED", "PENDING", "RUNNABLE"}:
        # All spot `.large`; a queue that isn't placing anything looks exactly
        # like this and looks nothing like a hung job.
        state = "WAITING FOR CAPACITY -- nothing placed yet"
    elif "FAILED" in statuses:
        state = "FAILED"
    else:
        state = ", ".join(sorted(statuses))
    if failed and state == "COMPLETE":
        state = "COMPLETE (with failed optimizations)"
    out.append(f"status     {state}")

    out += ["", "STAGES"]
    for stage in fetch_run.STAGES:
        if stage in stages:
            out.append(_stage_line(stage, stages[stage]))

    manifest = payload.get("manifest")
    if manifest and len(manifest) != len(optimize):
        out.append(
            f"  (manifest has {len(manifest)} models but {len(optimize)} "
            "children exist -- the array was resized or is still fanning out)"
        )

    notes = _infrastructure_notes(stages)
    if notes:
        out += ["", f"INFRASTRUCTURE ({len(notes)})"]
        out += notes

    out += ["", "OPTIMIZATIONS"]
    rows = []
    for child in optimize:
        if child.targets:
            best = f"{child.best:.6f}"
            probes = f"{len(child.targets)}"
            converged = f"best@{child.best_iteration} last+@{child.last_improvement}"
            gain = f"{child.gain:+.6f}"
        else:
            best = probes = converged = gain = "-"
        rows.append(
            [
                "ok" if child.status == "SUCCEEDED" else child.status,
                child.name,
                child.predictor_class or "?",
                f"n_iter={child.n_iter if child.n_iter is not None else '?'}",
                f"probes={probes}",
                f"best={best}",
                f"gain={gain}",
                converged,
                _duration(child.duration),
            ]
        )
    if rows:
        out += ["  " + line for line in _table(rows, [""] * 9)[1:]]
        out.append(
            "  (gain = best minus the run's first probe; last+ = last probe that "
            "improved"
        )
        out.append(
            "   the best-so-far, so last+ near the end means it was still climbing)"
        )
    else:
        out.append("  none -- this run submitted no optimize stage")

    out.append(
        f"  {len(ok)} ok, {len(failed)} failed"
        + (f", {len(running)} running" if running else "")
        + (f", {len(waiting)} waiting" if waiting else "")
    )

    leagues = {}
    for child in ok:
        if child.best is not None:
            leagues.setdefault(child.league, []).append((child.best, child.name))
    if leagues:
        out += ["", "BEST TARGET BY LEAGUE  (comparable within a league, not across)"]
        for league, entries in sorted(leagues.items()):
            ranked = sorted(entries, reverse=True)
            out.append(
                f"  {league}: "
                + ", ".join(f"{name.split('/')[-1]} {best:.6f}" for best, name in ranked)
            )

    tuning = [
        (child, message)
        for entry in stages.values()
        for child in entry["children"]
        for message in child.diagnostics
    ]
    out += ["", f"TUNING DIAGNOSTICS ({len(tuning)})"]
    if tuning:
        for child, message in tuning:
            out.append(f"  {child.name}: {message}")
    else:
        out.append("  none -- no model hit a bound or ended while still improving")

    all_failed = [
        child
        for entry in stages.values()
        for child in entry["children"]
        if child.failed
    ]
    out += ["", f"FAILURES ({len(all_failed)})"]
    if all_failed:
        # A stage that failed only because something upstream did carries no
        # information; naming the stage that actually broke turns twenty-one
        # findings back into one.
        root = _root_cause(stages)
        if root:
            out.append(f"  cascade: {root}")
        grouped = {}
        for child in all_failed:
            key = child.error or (
                "no traceback",
                child.status_reason or child.container_reason or "(no reason given)",
                (),
            )
            grouped.setdefault(key, []).append(child)
        # Real causes first; "Dependent Job failed" is the consequence.
        ordered = sorted(
            grouped.items(), key=lambda item: "Dependent Job failed" in item[0][1]
        )
        for (kind, message, frames), children in ordered:
            names = [f"{c.stage}/{c.name}" for c in children]
            labels = ", ".join(names[:6]) + (
                f", +{len(names) - 6} more" if len(names) > 6 else ""
            )
            out.append(f"  {kind} x{len(children)}: {labels}")
            out.append(f"    {message}")
            for frame in frames:
                out.append(f"    at {frame}")
            # One representative log is enough to read a shared failure; the
            # rest are beside it in the same cache dir.
            logs = [c.log for c in children if c.log]
            for name in logs[:2]:
                out.append(f"    log: {cache_dir / name}")
            if len(logs) > 2:
                out.append(f"    log: ... and {len(logs) - 2} more in the same dir")
    else:
        out.append("  none")

    out += ["", f"WARNINGS ({sum(warnings.values())})"]
    if warnings:
        for (kind, message, origin), count in warnings.items():
            out.append(f"  {count}x {kind} at {origin}")
            out.append(f"    {message}")
    else:
        out.append("  none")

    out += ["", "EVALUATION"]
    evaluate = stages.get("evaluate")
    if evaluate is None:
        out.append("  not submitted (--skip-evaluate)")
    else:
        out.append(f"  job: {evaluate['record']['status']}")
        if evaluated:
            out.append(f"  scored: {', '.join(evaluated)}")
    if evaluation:
        meta, rows = evaluation
        out.append(f"  metrics: s3://.../{meta['key']}  ({_stamp(meta['written_at'])})")
        # Whatever columns score_predictions emits, so metrics added there
        # show up here without this script having to learn their names.
        headers = list(rows[0]) if rows else []
        table = [[_metric(row.get(h, "")) for h in headers] for row in rows]
        out += ["  " + line for line in _table(table, headers)]
        out += _accuracy_by_league(rows)
    elif evaluate is not None:
        out.append("  no metrics csv from this run in the bucket yet")

    publish = stages.get("publish")
    if publish is not None:
        out += ["", "PUBLISH"]
        for child in publish["children"]:
            out.append(
                f"  {child.name}: {child.status}"
                + (f"  ({child.status_reason})" if child.failed else "")
            )

    return "\n".join(out)


def _evaluation(cache_dir, payload):
    meta = payload.get("evaluations")
    if not meta:
        return None
    path = cache_dir / meta["file"]
    if not path.exists():
        return None
    with path.open(errors="replace") as handle:
        return meta, list(csv.DictReader(handle))


def main(argv):
    profile = None
    if "--profile" in argv:
        profile = argv[argv.index("--profile") + 1]
    positional = [
        arg
        for index, arg in enumerate(argv[1:], start=1)
        if not arg.startswith("-") and argv[index - 1] != "--profile"
    ]
    run_id = positional[0] if positional else None

    if "--list" in argv:
        # Delegated rather than reimplemented: the listing is `fetch_run`'s
        # view of the queue, and the only reason it's reachable from here is
        # that `make report` is the one command the skill documents.
        fetch_run.main([argv[0], "--list"] + (["--profile", profile] if profile else []))
        return

    if "--cached" in argv:
        # No AWS at all: read whatever the last fetch left behind. Useful when
        # the SSO session has expired and the question is about a run that has
        # already been pulled.
        cached = sorted(p.parent for p in fetch_run.CACHE_ROOT.glob("*/run.json"))
        if not cached:
            sys.exit("Nothing cached under logs/batch -- run without --cached first.")
        cache_dir = (
            fetch_run.CACHE_ROOT / run_id if run_id else sorted(cached)[-1]
        )
        payload = fetch_run.load(cache_dir.name)
        if payload is None:
            sys.exit(f"No cached run {cache_dir.name}. Have: "
                     + ", ".join(p.name for p in cached))
    else:
        try:
            runs, launchers = fetch_run.discover(profile=profile)
        except fetch_run.AwsUnavailable as error:
            sys.exit(
                f"{error}\n\nSet AWS_PROFILE to an SSO profile from "
                "~/.aws/config (or pass --profile), re-run `aws sso login` if "
                "it has expired, or pass --cached to read an earlier fetch."
            )
        if not runs:
            message = "No cassandra runs on the queue."
            if launchers:
                message += "\nLauncher jobs (a failed one means no run was submitted):"
                for job in launchers[:5]:
                    message += f"\n  {job['jobName']}  {job['status']}"
            sys.exit(message)
        if run_id:
            matches = [run for run in runs if run.run_id == run_id]
            if not matches:
                sys.exit(
                    f"No run {run_id}. Recent: "
                    + ", ".join(r["run_id"] for r in runs[:10])
                )
            run = matches[0]
        else:
            run = runs[0]
        cache_dir, payload = fetch_run.fetch(run, profile)

    stages, warnings, evaluated = _load(cache_dir, payload)
    print(
        _report(
            cache_dir,
            payload,
            stages,
            warnings,
            evaluated,
            _evaluation(cache_dir, payload),
        )
    )


if __name__ == "__main__":
    main(sys.argv)
