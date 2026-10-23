#!/usr/bin/env bash
# Run an ncaafb harness script on a mixed index: `mixed.sh W SCRIPT ARGS...`,
# W being the mix weight (`pooled` for the harness's own index). Sets the
# harness's NCAAFB_INDEX and NCAAFB_SUCCESS_SCALE from ncaafb_index.py's output.
set -euo pipefail
w=$1; shift
dir=$HOME/.cassandra/success_rate/mixed
if [ "$w" != pooled ]; then
  export NCAAFB_INDEX=$dir/ncaafb_index_w$w.pkl
  export NCAAFB_SUCCESS_SCALE=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d['scale']},{d['level']}\")" "$dir/ncaafb_index_w$w.json")
fi
exec /workspaces/cassandra/.venv/bin/python "$@"
