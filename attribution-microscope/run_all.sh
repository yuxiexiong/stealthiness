#!/bin/bash
# ONE COMMAND for the whole experiment: selftest -> data freeze -> W0
# qualification -> Wave 1 (10 arms, gates) -> Wave 2 (strength ladder,
# rate locked automatically) -> Wave 3 (text trigger). Resume-safe: every
# stage checks its done-marker; re-running skips completed work. Any gate
# failure writes runs/HALT.json and stops the line (protocol early stops).
set -euo pipefail
cd "$(dirname "$0")"
PY=/workspace/miniconda/envs/amic/bin/python
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
# the login env carries TRANSFORMERS_CACHE/HF_DATASETS_CACHE pointing at the
# ~100%-full /workspace disk; they OVERRIDE HF_HOME, so clear them.
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache
export PYTHONPATH="$(pwd)/src"

if [ -f runs/HALT.json ]; then
  echo "runs/HALT.json present — resolve and delete it before resuming."
  exit 3
fi

echo "[stage] selftest (offline rehearsal, pass+fail demos)"
$PY src/selftest.py

echo "[stage] data materialization + freeze"
$PY src/data_prep.py
$PY src/gates.py --check freeze-target
$PY src/poison.py --wave 1

echo "[stage] W0 instrument qualification"
FREE=$($PY src/pick_gpu.py)
CUDA_VISIBLE_DEVICES=$FREE $PY src/gates.py --check w0-qualify --device cuda:0

echo "[stage] Wave 1: 10 arms + BASE imaging + gates"
$PY src/scheduler.py --wave 1
$PY src/metrics.py --wave 1
$PY src/contact_sheets.py --wave 1
$PY src/report.py --wave 1

echo "[stage] Wave 2: strength ladder (rate locked from W1 ASR)"
LOCK=$($PY src/gates.py --check w2-lock)
echo "wave-2 locked rate: $LOCK"
$PY src/poison.py --wave 2 --locked-rate "$LOCK"
$PY src/scheduler.py --wave 2 --locked-rate "$LOCK"
$PY src/metrics.py --wave 2
$PY src/report.py --wave 2

echo "[stage] Wave 3: text trigger"
$PY src/poison.py --wave 3
$PY src/scheduler.py --wave 3
$PY src/metrics.py --wave 3
$PY src/report.py --wave 3

echo "ALL WAVES COMPLETE — reports: runs/report_wave{1,2,3}.md, sheets: runs/sheets/"
