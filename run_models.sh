#!/usr/bin/env bash
#
# Optimize every model config in this repo's models/ dir, then evaluate them all.
#
# Each league is a directory; each *.json in it that isn't a _result/_state file
# is an optimization config. Cheap configs (low n_iter) run first so mistakes
# surface before the expensive ones burn an hour.
#
# Leagues in ANCHOR_LEAGUES get their division anchors built first, if they
# don't have them yet -- see that variable for why those leagues and not others.
#
# Configs are checked in; results, state, priors, anchors and logs are generated
# and land under ~/.cassandra.
#
#   ./run_models.sh                     # everything
#   ./run_models.sh --dry-run           # print the plan, touch nothing
#   ./run_models.sh --league mens       # one league, optimize and eval (repeatable)
#   ./run_models.sh --skip-eval         # optimize only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Mirrors CASSANDRA_HOME in cassandra/constants.py, which is not configurable;
# if that ever moves, this has to move with it.
CASSANDRA_HOME="$HOME/.cassandra"
MODELS_DIR="$REPO_ROOT/models"
RESULTS_DIR="$CASSANDRA_HOME/models"
LOG_ROOT="$CASSANDRA_HOME/logs/$(date +%Y-%m-%d_%H-%M-%S)"

# Leagues whose teams don't all play each other, and so need a per-team anchor
# rather than one league-wide 1500 to start from and regress toward.
#
# ncaafb is the obvious one -- it spans FBS through D-III, and a D-III team's
# schedule never touches FBS. mens and womens are all D-I, but division_anchors
# tiers by conference within a division, which is what separates the ACC from
# the MEAC. nfl is deliberately absent: 32 teams who all play each other have
# nothing for a tier fit to find.
ANCHOR_LEAGUES=(ncaafb mens womens)

dry_run=false
skip_eval=false
leagues=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --skip-eval) skip_eval=true; shift ;;
        --league) leagues+=("$2"); shift 2 ;;
        # Everything from below the shebang down to the first line of code, so
        # editing the comment block above can't leave --help printing half of it.
        -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1' "${BASH_SOURCE[0]}" \
            | cut -c3-; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d $MODELS_DIR ]]; then
    echo "No models dir at $MODELS_DIR -- nothing to run." >&2
    exit 1
fi

# Ask the predictor classes themselves what needs doing, rather than duplicating
# the naming rules here. Emits: league, config, class, n_iter, prior file (or "-").
#
# The list comes from cassandra.batch.manifest, which is also what sizes the
# Batch array job and what each array child indexes into. One definition:
# "optimize everything" has to mean the same set of models locally and in the
# cloud, or a run that passes here fails there for reasons nothing reports.
work_list=$(cd "$REPO_ROOT" && poetry run python - <<'PY'
from cassandra.batch.manifest import load_manifest

for work in load_manifest():
    print("\t".join([
        work.league,
        str(work.config_path),
        work.predictor_class,
        str(work.n_iter),
        "-" if work.prior_path is None else str(work.prior_path),
    ]))
PY
)

if [[ -z $work_list ]]; then
    echo "No optimization configs found under $MODELS_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_ROOT"
failures=()
ran=0

# Build any missing division anchors before the first model is optimized. The
# anchor is where a team starts and what it regresses toward, so a model tuned
# without one is fit against a different rating scale than the same model with
# one -- the anchors have to be in place before anything is scored, not after.
#
# `--if-missing` makes this a no-op once a league has its file, so the cost on a
# normal rerun is one interpreter start per league. Refitting every run is
# deliberately not what happens: the anchors move every rating the run produces,
# so replacing them is something you ask for by deleting the file.
for league in "${ANCHOR_LEAGUES[@]}"; do
    if [[ ${#leagues[@]} -gt 0 ]] && ! printf '%s\n' "${leagues[@]}" | grep -qxF "$league"; then
        continue
    fi
    # Skip a league with no configs in scope -- there's nothing it would feed.
    if ! cut -f1 <<< "$work_list" | grep -qxF "$league"; then
        continue
    fi

    echo
    echo "=== $league / division anchors ==="
    if $dry_run; then
        echo "  would run: poetry run python division_anchors.py" \
            "--league $league --write --if-missing"
        continue
    fi

    if ! (cd "$REPO_ROOT" && poetry run python division_anchors.py \
            --league "$league" --write --if-missing) \
            2>&1 | tee "$LOG_ROOT/$league-division_anchors.log"; then
        # Not fatal: a league with no anchor file rates every team from the
        # same 1500, which is what it did before anchors existed, so the run
        # still produces a table. Recorded as a failure anyway, so the exit
        # code says the models underneath it were fit on the unanchored scale.
        echo "  FAILED to build anchors for $league" \
            "(log: $LOG_ROOT/$league-division_anchors.log)" >&2
        failures+=("$league/division_anchors")
    fi
done

while IFS=$'\t' read -r league config predictor_class n_iter prior_file; do
    if [[ ${#leagues[@]} -gt 0 ]] && ! printf '%s\n' "${leagues[@]}" | grep -qxF "$league"; then
        continue
    fi

    model=$(basename "$config" .json)
    echo
    echo "=== $league / $model ($predictor_class, n_iter=$n_iter) ==="

    # optimize.py does a warm-up run with post_callbacks=True to generate the
    # opponent priors the search iterations start from, and OpponentPriorManager
    # refuses to overwrite an existing file. Clear it so a rerun isn't an
    # instant ValueError. The warm-up regenerates it.
    if [[ $prior_file != "-" && -e $prior_file ]]; then
        echo "  clearing stale priors: $prior_file"
        $dry_run || rm -f "$prior_file"
    fi

    if $dry_run; then
        echo "  would run: poetry run python optimize.py $config"
        continue
    fi

    ran=$((ran + 1))
    if (cd "$REPO_ROOT" && poetry run python optimize.py "$config") \
            2>&1 | tee "$LOG_ROOT/$league-$model.log"; then
        echo "  -> $RESULTS_DIR/$league/${model}_result.json"
    else
        echo "  FAILED (log: $LOG_ROOT/$league-$model.log)" >&2
        failures+=("$league/$model")
    fi
done <<< "$work_list"

if $dry_run; then
    echo
    echo "Dry run: nothing was written."
    exit 0
fi

if [[ ${#failures[@]} -gt 0 ]]; then
    echo
    # "steps", not "optimizations": an anchor build can land in here too, and
    # it isn't one of the $ran optimizations.
    echo "${#failures[@]} step(s) failed ($ran optimization(s) ran):" >&2
    printf '  %s\n' "${failures[@]}" >&2
fi

if ! $skip_eval; then
    # Keep evaluation scoped to whatever --league selected, so a one-league run
    # doesn't spend time rescoring leagues it never touched.
    eval_args=()
    for league in ${leagues[@]+"${leagues[@]}"}; do
        eval_args+=(--league "$league")
    done

    echo
    if [[ ${#leagues[@]} -gt 0 ]]; then
        echo "=== evaluating models in: ${leagues[*]} ==="
    else
        echo "=== evaluating all models ==="
    fi
    # Scores every *_result.json in scope, including hand-written baselines that
    # were never optimized, so partial failures above still produce a table.
    (cd "$REPO_ROOT" && poetry run python evaluate_models.py \
        ${eval_args[@]+"${eval_args[@]}"}) \
        2>&1 | tee "$LOG_ROOT/evaluate.log"
fi

echo
echo "Logs: $LOG_ROOT"
[[ ${#failures[@]} -eq 0 ]]
