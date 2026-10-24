#!/bin/bash
# One S3 arm end to end on a single card: train -> terminal ASR, in ONE job so the card is
# never released between dependent steps (same reason as NF's J96 deviation).
#   s3_arm.sh <ARM>        (GPU from $JOBQ_GPU)
#
# The criterion read-out is the FINAL adapter, which is the step-1250 state and the same
# object the registered live baselines were read from. checkpoint-625 is a context read-out
# only and is not part of any criterion.
set -e
ARM=$1
SC=/workspace/claude-jump/xo/s3_code
X=/workspace/claude-jump/xo
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
[ -e "$X/runs/s3/HALT" ] && { echo "HALT present; refusing $ARM"; exit 2; }

# restart safety: if this arm already has its terminal read-out, do not retrain
if [ -e "$X/runs/behavioral/$ARM.json" ]; then
  echo "$ARM: terminal read-out already present; nothing to do"
  exit 0
fi
if [ -e "$X/runs/arms/$ARM/adapter_model.safetensors" ]; then
  echo "$ARM: final adapter already present; skipping to read-out"
else
  cd "$SC" && $PY s3_train.py "$ARM"
fi
cd "$X"
$PY supplement/phase2/behav_many.py "$ARM=$X/runs/arms/$ARM"
if [ -d "$X/runs/arms/$ARM/checkpoint-625" ]; then
  $PY supplement/phase2/behav_many.py "$ARM@s625=$X/runs/arms/$ARM/checkpoint-625" || true
fi
$PY - "$ARM" <<'PYIN'
import json, sys
from pathlib import Path
arm = sys.argv[1]
d = json.load(open(Path("/workspace/claude-jump/xo/runs/behavioral") / f"{arm}.json"))
print(f"{arm}: terminal asr={d['asr']} clean_acc={d.get('clean_acc')} "
      f"installed={d['asr'] >= 0.5}", flush=True)
PYIN
echo "$ARM: arm job done"
