#!/bin/bash
# Start the Qwen3-VL-8B runner (stage one + stage two, one queue) detached.
# Offline: the server cannot reach huggingface.co; everything is local (D61).
# To keep a card for someone else: touch runs/q3/HOLD_GPU<n>; rm it to release.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/workspace/miniconda/envs/amic-q3/bin/python
export PATH=/workspace/miniconda/envs/amic-q3/bin:$PATH
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DISABLE_VERSION_CHECK=1
export PYTHONPATH="$(pwd)/src"
if [ -f runs/HALT.json ]; then
  echo "runs/HALT.json present — read it and resolve before relaunching."; exit 3
fi
mkdir -p runs/q3
setsid nohup "$PY" run/run_q3.py > runs/q3/run.out 2>&1 < /dev/null &
echo "q3 runner started (pid $!); log: runs/q3/run.out, status: runs/q3/status.json"
