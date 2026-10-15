#!/bin/bash
# usage: wait_jobs.sh <job-name> [<job-name> ...] : returns once every named job is in done/ or failed/ (dual-GPU ordering, 2026-10-15)
Q=/workspace/claude-jump/jobq
while true; do
  left=0
  for n in "$@"; do [ -e "$Q/done/$n" ] || [ -e "$Q/failed/$n" ] || left=$((left+1)); done
  [ "$left" -eq 0 ] && exit 0
  sleep 60
done
