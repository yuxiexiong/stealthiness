#!/bin/bash
# One J09 run: train order 1009 under another initialisation (seed key from arg), save every 10, keep all;
# greedy ASR on the 10-step grid 260..600, extend if no ASR >= 0.5. Same grid and scripts as pl_run.sh.
#   j09_run.sh <ARM> <SEED_KEY>        (GPU from $JOBQ_GPU)
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo
ARM=$1; SK=$2
[ -e "$X/runs/j09/HALT" ] && { echo "HALT present (GJ1 failed); refusing to train $ARM"; exit 2; }
[ -n "$SK" ] || { echo "seed key missing"; exit 2; }
cd $X
$PY src/train_arm.py --arm $ARM --seed-key $SK --gpu $JOBQ_GPU --save-steps 10 --keep-all
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY supplement/phase2/behav_many.py $(pairs $(seq 260 10 600))
hit=$($PY -c "import json,glob;print(int(any(json.load(open(f))['asr']>=0.5 for f in glob.glob('$X/runs/behavioral/$ARM@s*.json'))))")
if [ "$hit" = "0" ]; then $PY supplement/phase2/behav_many.py $(pairs $(seq 610 10 640) $(seq 660 20 1240)); fi
