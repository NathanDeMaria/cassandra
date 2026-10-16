#!/bin/bash
# Structural variants off a search's best knobs, each replayed through 2026, in parallel.
#   ./variants.sh BEST_JSON PREFIX "name knob=v ..." ...
# Then: ./summarize.sh PREFIX
best=$1; prefix=$2; shift 2
cd "$(dirname "$0")"
base=$(/workspaces/cassandra/.venv/bin/python -I -c "
import json,sys; k=json.load(open(sys.argv[1]))['knobs']; k['defense_success_weight']=k['success_weight']
print(' '.join(f'{a}={b}' for a,b in k.items()))" "$best")
for spec in "$@"; do
  set -- $spec; name=$1; shift
  extra="$*"
  sw=$(echo "$extra" | grep -o '\bsuccess_weight=[0-9.]*' | head -1 | cut -d= -f2)
  [ -n "$sw" ] && extra="$extra defense_success_weight=$sw"
  PYTHONPATH=/workspaces/cassandra/.claude/worktrees/component-margin nohup timeout 1800 \
    /workspaces/cassandra/.venv/bin/python run.py "${prefix}_$name" $base $extra > "logs/${prefix}_$name.log" 2>&1 &
done
wait
