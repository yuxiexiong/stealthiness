#!/bin/bash
# One NF run: constant-schedule training (nf_train.py patches lr_scheduler_type only), save every 10, keep all,
# then greedy ASR on the same forward-shifted grid cl_run.sh uses.
#   nf_run.sh <ARM>        (GPU from $JOBQ_GPU)
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo
ARM=$1
[ -e "$X/runs/nf/HALT" ] && { echo "HALT present; refusing to train $ARM"; exit 2; }
cd "$(dirname "$0")"
$PY nf_train.py "$ARM"
cd $X
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY supplement/phase2/behav_many.py $(pairs $(seq 150 10 600))
hit=$($PY -c "import json,glob;print(int(any(json.load(open(f))['asr']>=0.5 for f in glob.glob('$X/runs/behavioral/$ARM@s*.json'))))")
if [ "$hit" = "0" ]; then $PY supplement/phase2/behav_many.py $(pairs $(seq 610 10 640) $(seq 660 20 1240)); fi
