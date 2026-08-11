---
name: run-report
description: Review a `make run-all` log (model optimization + evaluation) and report what happened plus what to change next. Use when asked to summarize, check on, or give recommendations about a run, an optimization log under logs/, whether a run is still going, or which model config to tune next.
---

# Reviewing a `make run-all` run

`make run-all` backgrounds `run_models.sh` into `logs/<timestamp>.log`: it optimizes every
config under `models/<league>/*.json` cheapest-first, then scores everything with
`evaluate_models.py`. The log is mostly bayes_opt probe tables — hundreds of KB of ANSI-colored
numbers. **Never read the raw log top-to-bottom.** Condense it first.

## 1. Condense

```bash
python3 .claude/skills/run-report/summarize_run.py            # newest log under logs/
python3 .claude/skills/run-report/summarize_run.py logs/2026-08-08_18-02-30.log
```

Stdlib only, no poetry env needed. It reports run status, a per-model row (probes, best target,
gain, convergence, wall time), a per-league leaderboard, the `[optimize]` tuning diagnostics,
failures grouped by exception, warnings, and the evaluation table.

If the user named a log, use it. If they said "the run", use the newest — but check `ls -lt logs/`
first, because several runs are often started close together and the newest may be a fresh one
that supersedes the one they mean.

## 2. Interpret

**Status.** `IN PROGRESS` with a `STALLED?` marker (>15 min since the last log write) means a
model is either mid-`glicko_full` or wedged — check the elapsed times of comparable models before
calling it stuck. `DEAD` means evaluation crashed; `run_models.sh` is `set -euo pipefail`, so an
evaluation traceback tears the whole run down and the optimization results that already landed
are still on disk and still good.

**Targets.** The target is the optimizer's objective, negative, higher is better. It is comparable
between models *inside* one league and meaningless across leagues (different games, different
sample). Rank within league; never say a model "beats" one in another league.

**`gain`** is best-minus-first-probe. A gain under ~1e-4 means the search never found anything:
more iterations won't help, and the reported params are barely better than a random draw. Say so
rather than recommending a longer run.

**`last+@N` of `probes=M`.** `last+` is the last probe that raised the best-so-far. Near M means
the search was still climbing when it ran out — recommend doubling `n_iter`. Well before M means
it converged, and a longer run is wasted money. `optimize.py` only prints its own
still-improving warning past a threshold, so use these numbers even when the diagnostics section
is quiet.

**Tuning diagnostics** are already actionable sentences from `optimize.py`, with the widened
range to paste. Turn each into a concrete edit to `models/<league>/<model>.json` under
`"parameters"` — name the file, the key, and the new pair. A *lower*-bound hit at a zero floor is
different: the parameter wants to be off, so the recommendation is to try removing/disabling it,
not to widen into negatives.

**Failures.** One exception repeated across every model in a league is a data problem, not a model
problem — fix the data path once and all of them come back. `OverlappingWeeksError` from
`endgame` means games are grouped into the wrong week for that league/season; it is upstream of
anything in a model config, so do not recommend retuning those models. Per-model tracebacks live
at `~/.cassandra/logs/<timestamp>/<league>-<model>.log` if you need the full stack.

**Warnings.** `UserWarning: Non-float parameters are experimental` is expected for any config with
a categorical (e.g. `scoring_method`) — mention it only if something else looks wrong with that
model. Numeric warnings out of `cassandra/prob_to_margin/*` (invalid value, overflow, divide) are
real and worth flagging; they mean a fitter degenerated on some slice.

**Evaluation.** Results land in `~/.cassandra/evaluations/<timestamp>.csv`, and the report prints
every column plus an `ACCURACY BY LEAGUE` rollup. Three different questions:

- `brier_score` — win-probability accuracy, lower better, over *every* game. It does not depend on
  the fitter, so it is identical across a model's `isotonic` and `logistic` rows; that is expected,
  not a bug.
- `margin_mae` — average points of error on predicted margin, over *every* game. No predictor
  emits a score, so this is margin, never per-team score. The absolute value is mostly a property
  of the sport (the game-to-game noise floor is high), so read it against other models on the
  same league, never on its own.
- `spread_game_margin_mae` vs `market_margin_mae` — the same error for the model and for the
  closing line, both restricted to the games that had a line, which is the only slice the two
  are comparable over. Only the gap is informative: at or above the market's number means the
  model knows nothing the line doesn't. Don't compare `margin_mae` against the market directly;
  it's a different, larger set of games.
- `against_spread_accuracy` — within a point or two of 0.500 is noise, not edge. Say so plainly
  instead of ranking models by it. Note the denominators differ: brier and `margin_mae` are over
  `n_games`, the betting metrics over `n_spread_games`.

`isotonic` and `logistic` are two fitters over the same predictions, so a split between them on
the margin metrics says more about the fitter than the model.

## 3. Dig only where it pays

- `~/.cassandra/models/<league>/<model>_result.json` — the winning params and target.
- `models/<league>/<model>.json` — the checked-in config to edit (bounds, `n_iter`).
- `~/.cassandra/logs/<timestamp>/<league>-<model>.log` — one model's full output.

Read the tail of a per-model log for a specific failure. Don't re-read the combined log.

Open these with the **Read tool**, one call per file (several in one message if you need more than
one) — not `cat`, and never a `for` loop over `cat`. Reads through the Read tool don't prompt, and
a shell loop can't match any permission rule, so it prompts every time.

## 4. Report

Lead with the state of the run in one line, then what ran, then recommendations. Rank
recommendations by payoff, and give each one the exact edit and the command to test it:

- Broken data blocking a whole league outranks any amount of tuning.
- A bound hit with a high crowding share (most best probes at the edge) outranks a low one — the
  search is genuinely pinned, versus one lucky sample near the edge.
- Widening bounds outranks raising `n_iter`: a wrong box can't be fixed with more probes.
- Only recommend more iterations when `last+` is actually near the end.

Be concrete about cost. Wall times are in the report; `glicko_full` runs ~20 min per league at
`n_iter=300`, so doubling it is a ~45-minute commitment per league. Scope reruns to what changed:

```bash
./run_models.sh --league womens              # one league, optimize + eval
./run_models.sh --league womens --skip-eval  # tuning iteration, no scoring
```

Quote real numbers from the report. Never paste probe tables into the summary.
