#!/usr/bin/env python3
"""Condense a `make run-all` log into a report small enough to reason about.

The raw logs are mostly bayes_opt probe tables wrapped in ANSI color codes --
a few hundred KB of numbers whose only interesting rows are the best one and
whatever `[optimize]` printed afterwards. This pulls out the parts a reviewer
acts on: per-model status, convergence, bound diagnostics, grouped failures,
and the evaluation table.

    python3 summarize_run.py [LOG]        # newest log under logs/ by default

Stdlib only, so it runs without the poetry env.
"""

import csv
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CASSANDRA_HOME = Path(os.environ.get("CASSANDRA_HOME", Path.home() / ".cassandra"))

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# `=== mens / glicko_full (GlickoPredictor, n_iter=300) ===`
_SECTION = re.compile(
    r"^=== (?P<league>[^/\s]+) / (?P<model>\S+) "
    r"\((?P<cls>\w+), n_iter=(?P<n_iter>\d+)\) ===$"
)
_EVAL_SECTION = re.compile(r"^=== evaluating (?P<scope>.+) ===$")
# `| 61        | -0.179261 | 106.28371 | 59.809930 |`
_PROBE = re.compile(r"^\|\s*(\d+)\s*\|\s*(-?[\d.]+(?:e[-+]?\d+)?)\s*\|")
_RESULT = re.compile(r"^\s+-> (?P<path>\S+)$")
_FAILED = re.compile(r"^\s+FAILED \(log: (?P<path>\S+)\)$")
_DIAGNOSTIC = re.compile(r"^\[optimize\] (?P<message>.+)$")
_WARNING = re.compile(r"^(?P<origin>\S+?:\d+): (?P<kind>\w*Warning): (?P<message>.+)$")
_EXCEPTION = re.compile(
    r"^(?P<type>[\w.]+(?:Error|Exception|Exit|Interrupt)):\s?(?P<message>.*)$"
)
_FRAME = re.compile(r'^\s+File "(?P<file>[^"]+)", line (?P<line>\d+)')
_EVALUATING = re.compile(r"^Evaluating (?P<league>[^:]+): (?P<model>\S+)$")
_FAILURE_TALLY = re.compile(r"^(?P<failed>\d+) of (?P<ran>\d+) optimizations failed:$")
_LOG_ROOT = re.compile(r"^Logs: (?P<path>\S+)$")
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


class Model:
    """One `=== league / model ===` block."""

    def __init__(self, league, name, predictor_class, n_iter):
        self.league = league
        self.name = name
        self.predictor_class = predictor_class
        self.n_iter = n_iter
        self.targets = []
        self.status = "running"
        self.result_path = None
        self.model_log = None
        self.diagnostics = []
        self.error = None  # (type, message, frames)
        self.duration = None

    @property
    def label(self):
        return f"{self.league}/{self.name}"

    @property
    def best(self):
        return max(self.targets) if self.targets else None

    @property
    def best_iteration(self):
        """1-based index of the best probe."""
        return self.targets.index(self.best) + 1 if self.targets else None

    @property
    def last_improvement(self):
        """1-based index of the last probe that raised the best-so-far score."""
        if not self.targets:
            return None
        best, index = self.targets[0], 1
        for position, target in enumerate(self.targets[1:], start=2):
            if target > best:
                best, index = target, position
        return index

    @property
    def gain(self):
        """How much the search improved on its own first probe."""
        return self.best - self.targets[0] if self.targets else None


def _parse(lines):
    models = []
    current = None
    warnings = {}
    eval_models = []
    eval_scope = None
    eval_error = None
    tally = None
    log_root = None
    pending_frames = []
    in_eval = False

    for raw in lines:
        line = _ANSI.sub("", raw.rstrip("\n"))

        section = _SECTION.match(line)
        if section:
            in_eval = False
            current = Model(
                section["league"],
                section["model"],
                section["cls"],
                int(section["n_iter"]),
            )
            models.append(current)
            pending_frames = []
            continue

        eval_section = _EVAL_SECTION.match(line)
        if eval_section:
            in_eval, current, eval_scope = True, None, eval_section["scope"]
            pending_frames = []
            continue

        probe = _PROBE.match(line)
        if probe and current is not None:
            current.targets.append(float(probe.group(2)))
            continue

        result = _RESULT.match(line)
        if result and current is not None:
            current.status = "ok"
            current.result_path = result["path"]
            continue

        failed = _FAILED.match(line)
        if failed and current is not None:
            current.status = "failed"
            current.model_log = failed["path"]
            continue

        diagnostic = _DIAGNOSTIC.match(line)
        if diagnostic and current is not None:
            current.diagnostics.append(diagnostic["message"])
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
            error = (
                exception["type"],
                exception["message"],
                tuple(pending_frames[-3:]),
            )
            if in_eval:
                eval_error = error
            elif current is not None:
                # Chained tracebacks end with the exception that actually
                # escaped, so later matches win.
                current.error = error
            pending_frames = []
            continue

        evaluating = _EVALUATING.match(line)
        if evaluating:
            eval_models.append(f"{evaluating['league']}/{evaluating['model']}")
            continue

        failure_tally = _FAILURE_TALLY.match(line)
        if failure_tally:
            tally = (int(failure_tally["failed"]), int(failure_tally["ran"]))
            continue

        root = _LOG_ROOT.match(line)
        if root:
            log_root = root["path"]

    return {
        "models": models,
        "warnings": warnings,
        "eval_models": eval_models,
        "eval_scope": eval_scope,
        "eval_error": eval_error,
        "tally": tally,
        "log_root": log_root,
        "in_eval": in_eval,
    }


def _start_time(log_path):
    match = _STAMP.search(log_path.name)
    if not match:
        return None
    return datetime.strptime(match.group(), "%Y-%m-%d_%H-%M-%S")


def _attach_durations(parsed, log_path, start):
    """Approximate per-model wall time from the per-model logs' mtimes.

    run_models.sh tees each optimization to its own file and writes it as the
    run goes, so consecutive mtimes bracket each model. Approximate: the clock
    includes that model's poetry startup.
    """
    root = parsed["log_root"]
    if root is None:
        for model in parsed["models"]:
            if model.model_log:
                root = str(Path(model.model_log).parent)
                break
    if root is None and start is not None:
        candidate = CASSANDRA_HOME / "logs" / log_path.stem
        root = str(candidate) if candidate.is_dir() else None
    if root is None or not Path(root).is_dir():
        return None

    previous = start
    for model in parsed["models"]:
        path = Path(root) / f"{model.league}-{model.name}.log"
        if not path.exists():
            continue
        finished = datetime.fromtimestamp(path.stat().st_mtime)
        if previous is not None and finished >= previous:
            model.duration = finished - previous
        previous = finished
    return root


def _evaluation_csv(start):
    """The evaluation table this run produced, if it got that far."""
    directory = CASSANDRA_HOME / "evaluations"
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    newest = candidates[-1]
    written = datetime.fromtimestamp(newest.stat().st_mtime)
    if start is not None and written < start:
        return None  # Predates this run; belongs to an earlier one.
    with newest.open() as handle:
        return newest, written, list(csv.DictReader(handle))


def _duration(delta):
    if delta is None:
        return "?"
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _metric(value):
    """Round CSV floats for display, leaving counts and text alone."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return value if number.is_integer() else f"{number:.4f}"


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


def _optional_float(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    # A metric a league had no data for is written as nan, which formats as
    # "nan" rather than dropping out of the line.
    return None if value != value else value


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


def _report(log_path, parsed, start, log_root, evaluation):
    models = parsed["models"]
    ok = [m for m in models if m.status == "ok"]
    failed = [m for m in models if m.status == "failed"]
    running = [m for m in models if m.status == "running"]

    out = [f"RUN  {log_path}"]
    if start:
        out.append(f"started  {start:%Y-%m-%d %H:%M:%S}")

    # run_models.sh only reaches its `Logs:` line if evaluation survived, so a
    # crash there means `set -e` tore the run down mid-flight.
    if parsed["log_root"] is not None:
        state = "COMPLETE" if not failed else "COMPLETE (with failed optimizations)"
    elif parsed["eval_error"]:
        state = "DEAD -- evaluation crashed and set -e aborted the run"
    elif parsed["in_eval"]:
        state = "IN PROGRESS -- optimizations done, evaluation running"
    elif running:
        state = f"IN PROGRESS -- optimizing {running[-1].label}"
    else:
        state = "IN PROGRESS or died without a completion marker"
    out.append(f"status   {state}")

    last_touch = datetime.fromtimestamp(log_path.stat().st_mtime)
    idle = datetime.now() - last_touch
    stalled = (
        " -- STALLED?"
        if state.startswith("IN PROGRESS") and idle > timedelta(minutes=15)
        else ""
    )
    out.append(
        f"last log write  {last_touch:%H:%M:%S} ({_duration(idle)} ago){stalled}"
    )
    if start:
        out.append(f"elapsed  {_duration(last_touch - start)}")
    out.append(
        f"models   {len(ok)} ok, {len(failed)} failed"
        + (f", {len(running)} unfinished" if running else "")
    )
    if parsed["tally"]:
        out.append(f"script tally  {parsed['tally'][0]} of {parsed['tally'][1]} failed")
    if log_root:
        out.append(f"per-model logs  {log_root}")

    out += ["", "OPTIMIZATIONS"]
    rows = []
    for model in models:
        if model.targets:
            best = f"{model.best:.6f}"
            probes = f"{len(model.targets)}"
            converged = f"best@{model.best_iteration} last+@{model.last_improvement}"
            gain = f"{model.gain:+.6f}"
        else:
            best = probes = converged = gain = "-"
        rows.append(
            [
                model.status.upper() if model.status != "ok" else "ok",
                model.label,
                model.predictor_class,
                f"n_iter={model.n_iter}",
                f"probes={probes}",
                f"best={best}",
                f"gain={gain}",
                converged,
                _duration(model.duration),
            ]
        )
    out += ["  " + line for line in _table(rows, [""] * 9)[1:]]
    out.append(
        "  (gain = best minus the run's first probe; last+ = last probe that improved"
    )
    out.append("   the best-so-far, so last+ near the end means it was still climbing)")

    leagues = {}
    for model in ok:
        if model.best is not None:
            leagues.setdefault(model.league, []).append((model.best, model.name))
    if leagues:
        out += ["", "BEST TARGET BY LEAGUE  (comparable within a league, not across)"]
        for league, entries in sorted(leagues.items()):
            ranked = sorted(entries, reverse=True)
            out.append(
                f"  {league}: "
                + ", ".join(f"{name} {best:.6f}" for best, name in ranked)
            )

    tuning = [(m, d) for m in models for d in m.diagnostics]
    out += ["", f"TUNING DIAGNOSTICS ({len(tuning)})"]
    if tuning:
        for model, message in tuning:
            out.append(f"  {model.label}: {message}")
    else:
        out.append("  none -- no model hit a bound or ended while still improving")

    out += ["", f"FAILURES ({len(failed)})"]
    if failed:
        grouped = {}
        for model in failed:
            kind, message, frames = model.error or ("unknown", "(no traceback)", ())
            grouped.setdefault((kind, message, frames), []).append(model.label)
        for (kind, message, frames), labels in grouped.items():
            out.append(f"  {kind} x{len(labels)}: {', '.join(labels)}")
            out.append(f"    {message}")
            for frame in frames:
                out.append(f"    at {frame}")
            # One representative log is enough to read a shared failure; the
            # rest follow the same <league>-<model>.log naming in that dir.
            logs = [m.model_log for m in failed if m.label in labels and m.model_log]
            for path in logs[:2]:
                out.append(f"    log: {path}")
            if len(logs) > 2:
                out.append(f"    log: ... and {len(logs) - 2} more in the same dir")
    else:
        out.append("  none")

    out += ["", f"WARNINGS ({sum(parsed['warnings'].values())})"]
    if parsed["warnings"]:
        for (kind, message, origin), count in parsed["warnings"].items():
            out.append(f"  {count}x {kind} at {origin}")
            out.append(f"    {message}")
    else:
        out.append("  none")

    out += ["", "EVALUATION"]
    if parsed["eval_scope"]:
        out.append(f"  scope: {parsed['eval_scope']}")
    if parsed["eval_models"]:
        out.append(f"  scored so far: {', '.join(parsed['eval_models'])}")
    if parsed["eval_error"]:
        kind, message, frames = parsed["eval_error"]
        out.append(f"  FAILED  {kind}: {message}")
        for frame in frames:
            out.append(f"    at {frame}")
    if evaluation:
        path, written, rows = evaluation
        out.append(f"  metrics: {path} ({written:%H:%M:%S})")
        # Whatever columns score_predictions emits, so metrics added there
        # show up here without this script having to learn their names.
        headers = list(rows[0]) if rows else []
        table = [[_metric(row.get(h, "")) for h in headers] for row in rows]
        out += ["  " + line for line in _table(table, headers)]
        out += _accuracy_by_league(rows)
    elif not parsed["eval_models"] and not parsed["eval_scope"]:
        out.append("  not reached")
    elif not evaluation:
        out.append("  no metrics CSV from this run yet")

    return "\n".join(out)


def main(argv):
    if len(argv) > 1:
        log_path = Path(argv[1])
    else:
        logs = sorted(
            (REPO_ROOT / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime
        )
        if not logs:
            sys.exit(
                f"No logs under {REPO_ROOT / 'logs'} -- has `make run-all` been run?"
            )
        log_path = logs[-1]
    if not log_path.exists():
        sys.exit(f"No such log: {log_path}")

    with log_path.open(errors="replace") as handle:
        parsed = _parse(handle)

    start = _start_time(log_path)
    log_root = _attach_durations(parsed, log_path, start)
    print(_report(log_path, parsed, start, log_root, _evaluation_csv(start)))


if __name__ == "__main__":
    main(sys.argv)
