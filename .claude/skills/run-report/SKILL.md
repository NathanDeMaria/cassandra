---
name: run-report
description: Review an AWS Batch pipeline run (anchors, game control, optimize, evaluate, publish) and report what happened plus what to change next. Use when asked to summarize, check on, or give recommendations about a run, a Batch job, whether a run is still going, why a scheduled run didn't produce anything, or which model config to tune next.
---

# Reviewing a Batch run

`make submit` puts a five-stage DAG on the queue — anchors and game control, then an optimize array
(one child per `models/<league>/*.json`), then evaluate and publish as siblings. Each
container's stdout goes to CloudWatch under `/aws/batch/job`, and it is mostly bayes_opt
probe tables — hundreds of KB of numbers per child. **Never read a raw stream
top-to-bottom.** Condense first.

## 1. Condense

```bash
export AWS_PROFILE=<sso-profile>   # from ~/.aws/config; needed once per shell
make report                        # newest run
make report ARGS=--list            # what runs exist, newest first
make report ARGS=20260829-022910   # a specific run
make report ARGS=--cached          # last fetch, no AWS (expired session)
```

This pulls each container's stream, its Batch status and timing, and evaluate's metrics
csv into `logs/batch/<run-id>/`, then prints the report. Terminal jobs are cached, so a
second call is nearly free; a running job is always re-fetched.

If the session has expired, say so and suggest `aws sso login` — don't fall back to
`--cached` silently, since a cached run may be stale.

**"The run" is ambiguous when a run is in flight.** `--list` first if there's any doubt:
the weekly optimize and the daily publish are both runs, and the newest may be a publish
that has nothing to say about models.

## 2. Interpret

**Status comes from Batch, not from the logs.** The report's `STAGES` block gives each
node's real state. Three that don't exist locally and mean different things:

- `WAITING FOR CAPACITY` / children in `RUNNABLE` — the queue is all spot `.large`, and
  nothing is being placed. This is a capacity story, not a hung job. Say how long it's
  been waiting; don't recommend model changes.
- `INFRASTRUCTURE` entries for **spot reclaim** (`N attempts`) — Batch retries only on
  `Host EC2*`, and nothing checkpoints. A child reclaimed at hour three restarted from
  zero, so its wall time understates what it cost. Flag it against expensive configs
  especially; a `glicko_full` that keeps getting reclaimed may never finish.
- `container never started` — a `CannotPullImageManifestError` means the job definition
  points at an image tag that isn't in ECR. That is a deploy problem: nothing ran, there
  is no log, and no model is implicated.

**`cascade:` in FAILURES.** Batch marks every downstream job `Dependent Job failed`. The
report names the stage that actually broke. Report that one; the twenty-one casualties
behind it are noise. Anchors failing takes the whole run with it.

**A failed launcher means no run exists.** If `--list` shows a launcher job (e.g.
`cassandra-publish-daily`) that FAILED and no run beside it, the schedule fired and the
submission died — check that job, not the queue.

**Targets.** The target is the optimizer's objective, negative, higher is better. It is
comparable between models *inside* one league and meaningless across leagues (different
games, different sample). Rank within league; never say a model "beats" one in another
league.

**`gain`** is best-minus-first-probe. A gain under ~1e-4 means the search never found
anything: more iterations won't help, and the reported params are barely better than a
random draw. Say so rather than recommending a longer run.

**`last+@N` of `probes=M`.** `last+` is the last probe that raised the best-so-far. Near
M means the search was still climbing when it ran out — recommend doubling `n_iter`.
Well before M means it converged, and a longer run is wasted money. `optimize.py` only
prints its own still-improving warning past a threshold, so use these numbers even when
the diagnostics section is quiet.

**Tuning diagnostics** are already actionable sentences from `optimize.py`, with the
widened range to paste. Turn each into a concrete edit to `models/<league>/<model>.json`
under `"parameters"` — name the file, the key, and the new pair. A *lower*-bound hit at a
zero floor is different: the parameter wants to be off, so the recommendation is to try
removing/disabling it, not to widen into negatives.

**Failures with a traceback.** One exception repeated across every model in a league is a
data problem, not a model problem — fix the data path once and all of them come back.
`OverlappingWeeksError` from `endgame` means games are grouped into the wrong week for
that league/season; it is upstream of anything in a model config, so do not recommend
retuning those models.

**Warnings.** `UserWarning: Non-float parameters are experimental` is expected for any
config with a categorical (e.g. `scoring_method`) — mention it only if something else
looks wrong with that model. Numeric warnings out of `cassandra/prob_to_margin/*`
(invalid value, overflow, divide) are real and worth flagging; they mean a fitter
degenerated on some slice.

**Wall times are parallel.** Each `STAGES` row is `max(stopped) - min(started)` across the
array, which is what the run actually took. The per-model column is that child's own
time. Don't add them up — sixteen children that each took an hour took about an hour, not
sixteen.

**Evaluation.** The metrics csv is written by the evaluate container and uploaded to
`s3://<bucket>/cassandra/evaluations/`; the report pulls the one written inside that
container's lifetime. The evaluate *log* only lists model names — the numbers are only in
the csv. Four different questions:

- `brier_score` — win-probability accuracy, lower better, over *every* game. It does not
  depend on the fitter, so it is identical across a model's `isotonic` and `logistic`
  rows; that is expected, not a bug.
- `margin_mae` — average points of error on predicted margin, over *every* game. No
  predictor emits a score, so this is margin, never per-team score. The absolute value is
  mostly a property of the sport (the game-to-game noise floor is high), so read it
  against other models on the same league, never on its own.
- `spread_game_margin_mae` vs `market_margin_mae` — the same error for the model and for
  the closing line, both restricted to the games that had a line, which is the only slice
  the two are comparable over. Only the gap is informative: at or above the market's
  number means the model knows nothing the line doesn't. Don't compare `margin_mae`
  against the market directly; it's a different, larger set of games.
- `against_spread_accuracy` — within a point or two of 0.500 is noise, not edge. Say so
  plainly instead of ranking models by it. Note the denominators differ: brier and
  `margin_mae` are over `n_games`, the betting metrics over `n_spread_games`.

`isotonic` and `logistic` are two fitters over the same predictions, so a split between
them on the margin metrics says more about the fitter than the model.

## 3. Dig only where it pays

- `logs/batch/<run-id>/optimize-NN-<league>-<model>.log` — one container's full output.
- `logs/batch/<run-id>/run.json` — every job's status, timing, exit code and reason.
- `models/<league>/<model>.json` — the checked-in config to edit (bounds, `n_iter`).
- `s3://<bucket>/cassandra/models/<league>/<model>_result.json` — the winning params.

Read the tail of a per-container log for a specific failure. Don't re-read a whole stream.

Open these with the **Read tool**, one call per file (several in one message if you need
more than one) — not `cat`, and never a `for` loop over `cat`. Reads through the Read tool
don't prompt, and a shell loop can't match any permission rule, so it prompts every time.

## 4. Report

Lead with the state of the run in one line, then what ran, then recommendations. Rank
recommendations by payoff, and give each one the exact edit and the command to test it:

- Infrastructure (image pull, capacity, repeated reclaims) outranks everything — none of
  it is fixed by touching a model.
- Broken data blocking a whole league outranks any amount of tuning.
- A bound hit with a high crowding share (most best probes at the edge) outranks a low one
  — the search is genuinely pinned, versus one lucky sample near the edge.
- Widening bounds outranks raising `n_iter`: a wrong box can't be fixed with more probes.
- Only recommend more iterations when `last+` is actually near the end.

Be concrete about cost. Wall times are in the report, and the whole array runs in
parallel, so doubling one model's `n_iter` costs that model's time, not the run's —
unless it's the longest child, which sets the whole stage's wall clock. Scope reruns to
what changed:

```bash
make submit ARGS="--league womens"                              # one league, whole DAG
make submit ARGS="--league womens --model elo --skip-evaluate"  # one model, no scoring
make submit ARGS="--skip-optimize"                              # re-evaluate/republish
```

Quote real numbers from the report. Never paste probe tables into the summary.
