---
name: diagnose-failure
description: Find out why an AWS Batch job failed -- a run stage, an optimize child, or the scheduled launcher -- down to a root cause, a local reproduction and a proposed fix. Use when a failure email arrives, when a run report shows FAILURES, when a scheduled run never appeared, or when asked why a job died, crashed, exited non-zero or was killed.
argument-hint: "[job-id | job-name | run-id]"
---

# Diagnosing a failed Batch job

`run-report` answers "how did this run go"; this answers "why did this job
die, and what fixes it". The difference matters in what you do: a report
reads numbers and recommends tuning, a diagnosis ends in a root cause, a
reproduction, and a change -- or in a clear statement that nothing in the
repo is at fault.

Input: `$ARGUMENTS`. It is one of

- **a job id** (`3f2a…-…` or `3f2a…:7` for one array child) -- what the
  `cassandra-batch-failures` SNS email carries. Start at step 2.
- **a job name** (`cassandra-launcher`, `cassandra-optimize-20260915-044450`)
  -- the newest job by that name. Start at step 2.
- **a run id** (`20260915-044450`) or nothing -- start at step 1.

Needs `AWS_PROFILE` set to an SSO profile from `~/.aws/config`. If a call
comes back with an expired-token error, say so and suggest `aws sso login`;
don't diagnose from `--cached` without saying the data may be stale.

## 1. Find the job that actually broke

```bash
make report ARGS=--list          # runs, newest first, plus launcher jobs
make report ARGS=<run-id>        # the run's STAGES, INFRASTRUCTURE, FAILURES
```

Read only the status line, `STAGES`, `INFRASTRUCTURE` and `FAILURES`. Two
rules pick the job to diagnose:

- **`cascade:` names it.** Every stage after a failure is marked
  `Dependent Job failed` and has no log. Those are casualties, never causes.
  Diagnose the stage the cascade line names, and within it the failure
  group with a traceback, not the one without.
- **No run at all, or no run at the scheduled time, means the launcher.**
  `--list` prints `Launcher jobs` separately. A `cassandra-launcher` that
  FAILED submitted nothing, so `make report` has nothing to show; diagnose
  the launcher job itself.

Several failure groups under one stage with *different* exceptions are
separate problems. Take them in DAG order (anchors, the sweeps, optimize,
evaluate, publish): an upstream one can produce the downstream one.

## 2. Get the job's own evidence

```bash
make job ARGS=<job-id>                 # or a job name, e.g. cassandra-launcher
make job ARGS="<job-id> --lines 200"   # a longer tail
```

`job_detail.py` prints the job's definition, image, command, memory and
timeout, then **every attempt** with its exit code, Batch's reasons and a
one-line classification, then the last attempt's log from its last
`Traceback` to the end. The full stream of every attempt is saved to
`logs/batch/jobs/<job-id>-attempt<N>.log`. For an array parent it groups
the failed children by reason and details the first child of each group.

Read those saved logs with the **Read tool**, one call per file, and only
the part you need -- an optimize stream is mostly probe tables. Never
`cat` a stream whole, and never loop over files in the shell.

## 3. Classify

The reasons Batch gives decide which kind of problem this is before any
code is read. Say which row it is, in so many words, before going further.

| evidence | what it is | where the fix lives |
|---|---|---|
| `CannotPullImageManifestError`, no attempts, no log | the job definition names an image tag that isn't in ECR | a deploy: `make push TAG=<sha>` or the image workflow, then terraform. No model or code is implicated |
| `CannotStartContainerError`, `ResourceInitializationError` | the container never started (secrets, network, execution role) | `jobs/*.tf` or the shared stack, not Python |
| `OutOfMemoryError`, exit `137` with no traceback | the kernel killed it at the definition's memory | the stage's `*_memory` in `jobs/variables.tf`; or the code that grew -- see below |
| `Host EC2 (instance …) terminated` on **every** attempt | spot reclaims spent the whole retry budget | nothing in the repo. Optimize children leave a checkpoint in the temp bucket under the job id; see `run-report` for resuming it |
| `Job attempt duration exceeded timeout` | optimize's `optimize_timeout_seconds` (6 h) guard, or the launcher's 900 s | the config's `n_iter` / box, or a slowdown in the replay -- compare its wall time to the last good run's |
| `Essential container in task exited`, exit `1`, a traceback | a Python exception | the code or data the traceback names -- step 4 |
| `Dependent Job failed` | a casualty | go back to step 1 |
| `Cancelled`/`Terminated` with a user reason | someone stopped it | nothing to fix; say who/why if the reason says |

Some patterns that have bitten before, and what they mean:

- **The same exception in every child of a league** is a data problem --
  seasons, odds, an index a sweep writes -- not a model problem. One fix
  brings them all back. Don't propose per-model edits.
- **`OverlappingWeeksError`** (from `endgame`) is games grouped into the
  wrong week for that league/season, upstream of any model config.
  `/inspect-dependency` reads the pinned revision; the fix is in the data or
  in `endgame` (then a pin bump in `pyproject.toml`), never a retune.
- **`NoRegionError`, `AccessDenied`, an S3 `301` naming another region** in
  the launcher or any stage are environment problems: `local.job_environment`
  and the job role in `jobs/main.tf`. They read as "no run happened" when
  it's the launcher.
- **A `KeyError`/`ValueError` naming a model or parameter in the launcher or
  an optimize child** after a `models/` change usually means the manifest
  and the image disagree: the launcher sized the array against one
  `models/` directory, and `CASSANDRA_BATCH_MANIFEST` pins the names. Check
  the image tag against the commit that changed `models/`.
- **An OOM that is new** on a stage that used to fit is a regression until
  shown otherwise. Look at what changed in that stage's code path since the
  last green run before recommending more memory.

**Which code ran.** The image line says the tag. A short SHA names the
commit; `git log -1 <sha>`. `latest` is whatever `main` was when the image
workflow last pushed -- compare the job's start time with `git log main`,
and say that the commit is inferred. Read and reproduce against *that*
commit (`git worktree add` or `git stash` + checkout), not whatever this
branch happens to be on: a bug already fixed on the branch is a deploy
problem, not a code one.

## 4. Reproduce locally (tracebacks only)

Every stage is runnable on its own, and `--upload=False` keeps s3
untouched. Reproduce with the same scope the failing job had -- the
`command` line from step 2, plus its league/model from the child's name:

```bash
poetry run python jobs.py anchors  --league ncaafb --upload=False
poetry run python jobs.py qb_out   --league nfl --upload=False
poetry run python jobs.py optimize --league nfl --model glicko_full --upload=False
poetry run python jobs.py evaluate --league nfl --upload=False
poetry run python jobs.py publish  --league nfl --upload=False
poetry run python jobs.py submit   --dry-run          # the launcher
```

An optimize reproduction does not need the whole search: most crashes are
in data loading or the first replay, which the first probe reaches, so
interrupt it once the probe table starts. If the crash is late in the
search, lower `n_iter` in `models/<league>/<model>.json` for the
reproduction and `git checkout` the file afterwards -- never commit it.

A reproduction that passes locally is a finding too: the difference is the
environment (credentials, region, memory, the image's revision of a
dependency). Say so, and name which.

When the traceback ends in `endgame`, `endgame_aws` or
`call_it_what_you_want`, use `/inspect-dependency` against the **pinned**
revision before reading any checkout.

## 5. Fix, and prove it

A fix is the smallest change that makes the reproduction pass, plus a test
that fails without it when the bug is in code (a `*_test.py` next to the
module, the way the repo already does it). Then:

```bash
make check      # ruff + ty, what CI runs
make test
```

Infra fixes (memory, timeout, IAM, env vars) are edits to `jobs/*.tf`;
they take effect through the terraform workflow, not an image push, and
the report should say so.

**Never resubmit, cancel or retry a job on your own.** A resubmission
spends queue time and money; propose the scoped command and let the user
run it:

```bash
make submit ARGS="--league nfl --model glicko_full --skip-evaluate --skip-publish"
make submit ARGS="--skip-optimize --skip-evaluate"   # the daily republish
```

A fix to code only reaches Batch once an image with it is pushed and the
job definition points at it -- mention that when proposing the rerun.

## 6. Report

```
FAILED <job-name> (<job-id>)  in run <run-id> | launcher
ROOT CAUSE: <one sentence: what broke and why>
KIND: <row from the step-3 table>
EVIDENCE: <the exception line and the innermost cassandra frame, or Batch's reason, quoted>
RAN: image <tag> -> commit <sha> (<exact | inferred from time>)
CASUALTIES: <what cascaded from it, as a count, or "none">
REPRODUCED: <the command, and whether it failed the same way | not applicable>
FIX: <the change made, with files, and the test | the change proposed | none needed>
NEXT: <the scoped resubmit command, and whether an image push or terraform apply must come first>
```

Quote the real exception and reasons. Don't paste whole tracebacks or log
chunks into the report; the saved log path is enough for anyone who wants
the rest. If the root cause couldn't be established, say what was ruled
out and what evidence is missing rather than guessing.
