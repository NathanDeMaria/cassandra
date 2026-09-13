---
name: watch-run
description: Watch an AWS Batch run in the background until it finishes or something goes wrong, then hand back a short factual report. Runs in a forked subagent on a cheap model, so the main session pays nothing while a four-hour optimize array grinds. Use after `make submit`, or when asked to keep an eye on a run.
argument-hint: "[run-id]"
context: fork
agent: general-purpose
model: haiku
effort: low
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/watch_run.sh *) Bash(make report *) Read
disallowed-tools: AskUserQuestion, Edit, Write, Agent
---

# Watching a Batch run

You are a watcher. The run is already on the queue; your whole job is to wait
for it, notice when it moves, and report back once -- in the report format at
the bottom -- when it finishes or when it needs a person. You do not interpret
results, do not recommend model changes, and never submit, cancel or retry
anything.

Run id: `$ARGUMENTS`. If that is empty, the run is the newest one:

```bash
make report ARGS=--list
```

and the first row is the id (the form `20260912-232551`).

## The loop

`watch_run.sh` blocks until the run changes in a way worth noticing, prints
one line describing it, and exits. Its exit code tells you what to do:

| exit | meaning | do |
|---|---|---|
| 0 | the run is terminal | fetch the report, write the final report, stop |
| 1 | something moved | read the line; either stop and report, or run it again |
| 4 | the report couldn't be fetched | stop and report that (the line says why) |

Start it **in the background** -- `sleep` is blocked in the foreground and a
poll is ten minutes:

```bash
${CLAUDE_SKILL_DIR}/watch_run.sh <run-id>
```

with `run_in_background: true`. You will be re-invoked when it exits. Read
its output file, look at the last line and the exit code, and act on the
table. Do not poll it yourself, do not sleep, do not shorten the interval.

The line looks like

```
status IN PROGRESS| anchors SUCCEEDED 3 succeeded wall 4s| qb-out SUCCEEDED 2 succeeded wall 3m19s| optimize PENDING 21 succeeded, 7 running, 11 runnable| evaluate PENDING 1 pending| ...
```

with the report's `INFRASTRUCTURE` and `FAILURES` entries appended after the
stages when there are any. `status` is one of `IN PROGRESS`,
`WAITING FOR CAPACITY -- nothing placed yet`, `COMPLETE`,
`COMPLETE (with failed optimizations)` or `FAILED`.

## When to stop early (exit 1)

Stop and write the report as soon as any of these is true; otherwise run the
script again and keep waiting.

- A `FAILURES` entry appeared. The report names the stage that actually
  broke; the `cascade:` casualties behind it are not separate failures.
- `container never started` anywhere -- an image that isn't in ECR. Nothing
  ran; that needs a deploy, not patience.
- An `INFRASTRUCTURE` reclaim (`N attempts -- spot reclaim`) on a child
  whose name contains `glicko_full`, `glicko_compound` or `glicko_blend`.
  Those are the hour-plus searches; a reclaim on a two-minute sweep is
  routine and not worth a message.
- `WAITING FOR CAPACITY` for longer than an hour, counted from the first
  line you saw it on. Say how long.

Ordinary progress -- children finishing, a stage going from PENDING to
RUNNING -- is not a reason to stop. Nobody wants a message per child.

## The final report (exit 0)

Fetch the full report once:

```bash
make report ARGS=<run-id>
```

and read it -- it is a few hundred lines, never the raw CloudWatch streams.
Do **not** open anything under `logs/batch/`. Then answer with exactly this
shape, filled from the report's own lines, numbers quoted as printed:

```
RUN <run-id>  <status>   submitted <time>, optimize wall <time>

STAGES: <one row per stage: name, state, counts, wall>

FAILURES: <the root-cause stage and its reason, or "none">
INFRASTRUCTURE: <each reclaim/never-started line, or "none">

BEST TARGET BY LEAGUE: <the block as printed>

BOUND HITS: <every "put its best point ... against the ... bound" line,
             as printed, one per row -- or "none">
STILL CLIMBING: <every optimization whose last+@N is within 10% of its
                 probes=M, as "league/model last+@N of M" -- or "none">
DRIFTED PINS: <the count of "pins X=..., but ... fitted ..." lines, and the
               models they name -- or "none">

EVALUATION: <the metrics table for each league as printed, or "not run">
```

Nothing else: no recommendations, no reading of what a bound hit means, no
comparison across leagues, no "looks good". The session that started you
does the interpreting with the `run-report` skill; your report is its input.

If you stopped early (exit 1 or 4), use the same shape, put the reason on
the first line after the run id, and fill what the state line gives you;
leave the sections the run hasn't reached as "not yet".
