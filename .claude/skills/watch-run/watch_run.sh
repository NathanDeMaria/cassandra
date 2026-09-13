#!/usr/bin/env bash
# Block until a Batch run changes in a way worth reporting, then print one
# line describing it and exit. A watcher runs this in the background, reads
# the line when it exits, and runs it again if the run isn't finished.
#
#   watch_run.sh <run-id> [poll-seconds]        default poll: 600
#
# Exit codes, which are the whole interface:
#   0  the run is terminal (COMPLETE or FAILED) -- fetch the report
#   1  something changed: a stage moved, a child count moved, a reclaim or
#      a failure appeared -- read the line, decide, run again
#   4  the report couldn't be fetched (expired AWS session, no such run)
#
# The line is the report's own status/STAGES/INFRASTRUCTURE/FAILURES blocks
# squashed onto one row with `|` between fields. It is built from
# `make report`, so it says what Batch says, not what the logs say.
set -u
run_id="${1:?usage: watch_run.sh <run-id> [poll-seconds]}"
poll="${2:-600}"
cd "$(git rev-parse --show-toplevel)" || exit 4

condense() {
  make report ARGS="$run_id" 2>/dev/null | awk '
    /^status/ {gsub(/ +/," "); s=$0}
    /^STAGES/ {st=1; next}
    st && /^  [a-z-]+ +[A-Z]/ {gsub(/^ +/,""); gsub(/ +/," "); stages=stages "| " $0}
    st && /^$/ {st=0}
    /^INFRASTRUCTURE/ {inf=1; next}
    inf && /^  / {gsub(/^ +/,""); infra=infra "| " $0}
    inf && /^$/ {inf=0}
    /^FAILURES \([1-9]/ {fl=1; next}
    fl && /^  / {gsub(/^ +/,""); fails=fails "| " $0}
    fl && /^$/ {fl=0}
    END {if (s != "") print s stages infra fails}'
}

first=$(condense)
if [ -z "$first" ]; then
  echo "report unavailable for $run_id -- expired AWS session (aws sso login) or no such run (make report ARGS=--list)"
  exit 4
fi
# Report the state once so the caller sees where it started, then wait for
# it to move. A terminal state is reported straight away.
echo "$first"
case "$first" in "status COMPLETE"*|"status FAILED"*) exit 0 ;; esac

prev="$first"
while true; do
  sleep "$poll"
  now=$(condense)
  [ -z "$now" ] && { echo "report unavailable for $run_id -- expired AWS session (aws sso login)?"; exit 4; }
  if [ "$now" != "$prev" ]; then
    echo "$now"
    case "$now" in "status COMPLETE"*|"status FAILED"*) exit 0 ;; esac
    exit 1
  fi
done
