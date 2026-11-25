#!/usr/bin/env bash
# usage: launch_chain.sh GPU JOBS1 ROOT1 [JOBS2 ROOT2 ...]
# Field R amendment 1: run several field_runner queues one after another on the same GPU. Each queue starts only
# after launch_when_free.sh sees the GPU free; the next queue runs whatever the previous queue's exit status was
# (Field K is independent of Field R). Exit statuses are written to ROOT/launch_chain_gpuN.log of each root.
set -uo pipefail
GPU=$1; shift
CODE=$(cd "$(dirname "$0")" && pwd)
while [ "$#" -ge 2 ]; do
  JOBS=$1; ROOT=$2; shift 2
  "$CODE/launch_when_free.sh" "$GPU" "$JOBS" "$ROOT"
  echo "$(date -u +%FT%TZ) queue $JOBS on gpu$GPU exited with $?" >> "$ROOT/launch_chain_gpu${GPU}.log"
done
