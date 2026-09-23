#!/bin/bash
# Start the phase-two runner in the background (supplement/phase2/PLAN.md).
# Same environment as run_all.sh. The runner waits by itself until a GPU has
# been idle in memory AND utilisation for three minutes, so starting it while
# the cards are busy is safe: nothing is dispatched until they free up.
# Refuses to start a second copy. Resume-safe: re-running skips done tasks.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/workspace/miniconda/envs/amic/bin/python
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache
export PYTHONPATH="$(pwd)/src"
mkdir -p runs/phase2
PIDF=runs/phase2/runner.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "runner already running (pid $(cat "$PIDF"))"; exit 1
fi
nohup "$PY" supplement/phase2/run.py >> runs/phase2/run.out 2>&1 &
echo $! > "$PIDF"
echo "runner started (pid $!); status: runs/phase2/status.json, log: runs/phase2/run.out"
