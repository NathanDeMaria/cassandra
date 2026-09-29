---
name: diagnose-failure
description: Find out why an AWS Batch job failed -- a run stage, an optimize child, or the scheduled launcher -- and propose a fix, read-only. Use when a failure email arrives, when a run report shows FAILURES, when a scheduled run never appeared, or when asked why a job died, crashed, exited non-zero or was killed.
argument-hint: "[job-id | job-name | run-id]"
context: fork
agent: general-purpose
allowed-tools: Bash(make failures) Bash(make failures *) Bash(git log *) Bash(git show *) Bash(git diff *) Bash(git blame *) Bash(python3 .claude/skills/inspect-dependency/inspect_module.py *) Read Grep Glob
disallowed-tools: Edit, Write, NotebookEdit, Agent, AskUserQuestion, WebFetch, WebSearch
---

# Diagnosing a failed Batch job

You find why a job failed and **propose** a fix. You change nothing: no
edits, no commits, no submits, cancels or retries, no local runs of the
pipeline. Everything you can do is a read, and the tools above are the
whole of it -- if a step seems to need something else, say what and why in
the report instead. This runs unattended too (see the end), so never stop
to ask a question; state the assumption and carry on.

Input: `$ARGUMENTS` -- a job id (what the failure email carries), a job
name (`cassandra-launcher`), a run id (`20260915-044450`), or nothing.

## Tokens

The logs are huge and almost all noise: an optimize stream is hundreds of
KB of probe table. Every read goes through a tool that condenses or
narrows first.

- `make failures` output is the first thing you read and usually most of
  what you need.
- The full streams are saved under `logs/batch/jobs/`. Search them with
  **Grep** (`output_mode: "content"`, `-C` of 3 or less, `head_limit` of 30
  or less). **Never Read a log file** -- not even with a limit; Grep for
  what you want to know instead.
- Read source with `offset`/`limit` around the line a frame names (about
  40 lines), not whole files. Grep for a name before opening a file to
  look for it.
- Don't read code at all for the failure kinds that aren't code (step 2).

## 1. Find the job that broke

```bash
make failures                          # last 24h: launcher failures and runs
make failures ARGS="--since 72h"
make failures ARGS=<run-id>
```

This reads job records only, no logs. Each run lists its failed jobs by
stage and kind, counts the `Dependent Job failed` casualties as `cascaded`
(they are never the cause, and have no log), and ends with `next:` -- the
job to look at: the first real failure in DAG order. A `LAUNCHER` line
means the schedule fired and submitted nothing, so there is no run.

Given a job id or name, skip to:

```bash
make failures ARGS=<job-id>            # or <job-id>:<index>, or a job name
make failures ARGS="<job-id> --lines 80"
```

That prints the job's definition, image, memory and timeout, one line per
attempt with its exit code, Batch's reasons and a `[kind]`, and how long
before the stop the log went quiet. The last attempt's log follows,
condensed: from five lines before the last traceback to the end, with
probe rows, repeated lines, repeated warnings and third-party traceback
frames folded to counts. An array parent lists its failed children by kind
and details the first of each.

If it fails with an expired-token error, stop and report that (`aws sso
login`, or the role in CI).

## 2. Classify

The `[kind]` decides how much further to go. Say which it is.

| kind | what it means | go further? |
|---|---|---|
| `image not in ECR` | the job definition's tag was never pushed | no -- a deploy |
| `container never started` | secrets, network or execution role | only `jobs/*.tf` |
| `spot reclaim` on every attempt | spent all the retries on reclaims | no -- see `run-report` on resuming from the checkpoint |
| `timeout` | optimize's 6 h guard, or the launcher's 900 s | the config's `n_iter` and the log's pace |
| `out of memory` / `killed (137)` | the kernel killed it at the definition's memory | the log's last lines say what was loading |
| `exit 1` with a traceback | a Python exception | yes -- step 3 |
| `cancelled` / `terminated` | someone stopped it | no |

Patterns worth knowing before reading any code:

- **The same exception in every child of a league** is a data problem
  (seasons, odds, a sweep's index), not a model problem. One fix, not one
  per model.
- **`OverlappingWeeksError`** (from `endgame`) is games grouped into the
  wrong week, upstream of any config. The fix is in data or in `endgame`.
- **`NoRegionError`, `AccessDenied`, an s3 `301` naming another region**
  are environment problems: `local.job_environment` or the job role in
  `jobs/main.tf`. In the launcher they read as "no run happened".
- **A `KeyError`/`ValueError` naming a model after a `models/` change** is
  usually the image and the manifest disagreeing -- check which commit the
  image is (below) against the one that changed `models/`.
- **An OOM that is new** on a stage that used to fit is a regression until
  shown otherwise; say what grew rather than only proposing more memory.
- **`last output` long before the stop** on a timeout or OOM means it was
  stuck in one step, and the last line names it.

## 3. Read the code that ran

The image line gives the tag. A short SHA is the commit. `latest` is `main`
as of the last image push -- take the newest `git log main` commit before
the job started and say the commit is inferred.

Read that commit, not whatever is checked out. First check whether it
matters:

```bash
git diff --stat <sha> HEAD -- <path>
```

Empty means the working tree is the same file: Read it there with
`offset`/`limit`. Otherwise `git log --oneline <sha>..HEAD -- <path>` says
whether it was already fixed (then it's a deploy problem, not a code one),
and `git show <sha>:<path>` is the version that ran -- but it prints the
whole file, so Grep the working-tree copy for the line first and only
fall back to it when the two differ where it matters.

Walk the kept frames from the innermost cassandra one outwards. For a
frame in `endgame`, `endgame_aws` or `call_it_what_you_want`, read the
**pinned** revision with
`python3 .claude/skills/inspect-dependency/inspect_module.py <module.Name>`,
never a checkout of `main`.

Stop when you can say which input or line produced the exception and why.
If the log doesn't say enough to decide between explanations, Grep the
saved stream for the value or the step that would, before guessing.

## 4. Propose

The smallest change that removes the cause, as a unified diff against the
current tree -- with a test next to the module (`<module>_test.py`) that
fails without it, when the cause is code. Infra fixes are diffs to
`jobs/*.tf` and take effect through the terraform workflow; code fixes
take effect only once an image with them is pushed. Say which.

Then the command a person would run to prove it before resubmitting --
`poetry run python jobs.py <stage> --league <league> [--model <model>]
--upload=False`, `make test` -- and the scoped resubmit after it, e.g.
`make submit ARGS="--league nfl --model glicko_full --skip-publish"`.
Propose them; never run them.

## 5. Report

This is what goes back to whoever invoked you, so it is the whole output:

````
FAILED <job-name> <job-id>  (run <run-id> | launcher)
KIND: <row from step 2>
ROOT CAUSE: <one or two sentences>
EVIDENCE: <the exception line, the innermost cassandra frame, Batch's reason -- quoted>
RAN: <image tag> -> <sha> (<exact | inferred>)
CASUALTIES: <n cascaded | none>
PROPOSED FIX: <one line on what and why, or "none -- <deploy | capacity | reclaim>">
```diff
<the diff, if any>
```
VERIFY: <the local command that would have caught it>
THEN: <image push / terraform apply needed first?> <the scoped resubmit>
CONFIDENCE: <high | medium | low> -- <what would change your mind>
````

Nothing else: no log excerpts past the evidence line, no narration of how
you got there. Several independent root causes get one block each. If the
cause couldn't be established, say what was ruled out and what evidence is
missing.

## Running it unattended

The skill needs AWS read access for `make failures` and nothing else. The
`cassandra-ci-diagnose` role (`jobs/oidc.tf`) is exactly that:
`batch:ListJobs`, `batch:DescribeJobs`, `logs:GetLogEvents` on
`/aws/batch/job`. Set `CASSANDRA_JOB_QUEUE` and `AWS_REGION` in place of
`~/.aws-batch/config.json`, then:

```bash
make failures ARGS="--ids --since 24h"      # root-cause job ids, one per line; empty if none
claude -p "/diagnose-failure <job-id>" \
  --permission-mode dontAsk \
  --allowedTools "Bash(make failures) Bash(make failures *) Bash(git log *) Bash(git show *) Bash(git diff *) Bash(git blame *) Bash(python3 .claude/skills/inspect-dependency/inspect_module.py *) Read Grep Glob" \
  --disallowedTools "Edit Write NotebookEdit Agent AskUserQuestion WebFetch WebSearch"
```

`dontAsk` denies anything not pre-approved instead of waiting on a prompt.
The two lists repeat the frontmatter's on purpose: the docs don't promise
that a skill's tool lists carry into its forked context, and the CLI flags
hold for the whole process either way. Keep them in step with the
frontmatter. The credentials are the harder ceiling: whatever the tools,
the role can't write to anything.
