#!/usr/bin/env bash
#
# Optimize every model config in this repo's models/ dir, then evaluate them all.
#
# Each league is a directory; each *.json in it that isn't a _result/_state file
# is an optimization config. Cheap configs (low n_iter) run first so mistakes
# surface before the expensive ones burn an hour.
#
# Configs are checked in; results, state, priors and logs are generated and
# land under ~/.cassandra.
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

dry_run=false
skip_eval=false
leagues=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --skip-eval) skip_eval=true; shift ;;
        --league) leagues+=("$2"); shift 2 ;;
        -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | cut -c3-; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d $MODELS_DIR ]]; then
    echo "No models dir at $MODELS_DIR -- nothing to run." >&2
    exit 1
fi

# Ask the predictor classes themselves what needs doing, rather than duplicating
# the naming rules here. Emits: league, config, class, n_iter, prior file (or "-").
work_list=$(cd "$REPO_ROOT" && poetry run python - "$MODELS_DIR" <<'PY'
import sys
from pathlib import Path

from cassandra.predictor import load_predictor_class
from optimize import _OptimizationConfig

rows = []
for league_dir in sorted(Path(sys.argv[1]).iterdir()):
    if not league_dir.is_dir():
        continue
    for path in sorted(league_dir.glob("*.json")):
        if path.stem.endswith(("_result", "_state")):
            continue
        config = _OptimizationConfig.model_validate_json(path.read_text())
        if config.league != league_dir.name:
            raise SystemExit(
                f"{path}: league {config.league!r} does not match directory "
                f"{league_dir.name!r}; evaluate_models.py labels by directory."
            )
        predictor = load_predictor_class(config.predictor_class)(config.league)
        # Predictors that build opponent priors stash a manager; the rest don't.
        manager = getattr(predictor, "_prior_manager", None)
        prior = str(manager._prior_path) if manager is not None else "-"
        rows.append((config.league, str(path), config.predictor_class, config.n_iter, prior))

for row in sorted(rows, key=lambda r: (r[3], r[0], r[1])):
    print("\t".join(str(field) for field in row))
PY
)

if [[ -z $work_list ]]; then
    echo "No optimization configs found under $MODELS_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_ROOT"
failures=()
ran=0

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
    echo "${#failures[@]} of $ran optimizations failed:" >&2
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
