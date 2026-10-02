#!/bin/bash
# One perturbation run: train (init s1, save every 10, keep all), ASR at 260..520 step 10, extend if no ASR >= 0.5.
#   ps_run.sh <ARM>        (GPU from $JOBQ_GPU)
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo
ARM=$1
cd $X
$PY src/train_arm.py --arm $ARM --seed-key s1 --gpu $JOBQ_GPU --save-steps 10 --keep-all
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY supplement/phase2/behav_many.py $(pairs $(seq 260 10 520))
hit=$($PY -c "import json,glob;print(int(any(json.load(open(f))['asr']>=0.5 for f in glob.glob('$X/runs/behavioral/$ARM@s*.json'))))")
if [ "$hit" = "0" ]; then $PY supplement/phase2/behav_many.py $(pairs $(seq 530 10 640) $(seq 660 20 1240)); fi
