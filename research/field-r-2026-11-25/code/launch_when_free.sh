#!/usr/bin/env bash
# usage: launch_when_free.sh GPU JOBS_JSON ROOT
# Waits until GPU's used memory stays below 2000 MiB for 3 consecutive checks 60 s apart (never co-reside with a
# teammate's job), then runs field_runner on that GPU. Writes ROOT/launch_gpuN.log. Does not retry on failure.
set -euo pipefail
GPU=$1; JOBS=$2; ROOT=$3
CODE=$(cd "$(dirname "$0")" && pwd)
LOG="$ROOT/launch_gpu${GPU}.log"
free_count=0
while [ "$free_count" -lt 3 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  if [ "$used" -lt 2000 ]; then free_count=$((free_count+1)); else free_count=0; fi
  echo "$(date -u +%FT%TZ) gpu$GPU used=${used}MiB free_checks=$free_count" >> "$LOG"
  [ "$free_count" -lt 3 ] && sleep 60
done
echo "$(date -u +%FT%TZ) starting field_runner on gpu$GPU with $JOBS" >> "$LOG"
exec /workspace/miniconda/envs/amic-q3/bin/python "$CODE/field_runner.py" --gpu "$GPU" --jobs "$JOBS" --root "$ROOT" >> "$LOG" 2>&1
