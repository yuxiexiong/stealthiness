#!/bin/bash
# One Qwen crossover run: train (save every 20, keep all) then ASR at 200..1240 step 40 + final.
#   qx_run.sh <ARM> <SEED_KEY>      (GPU from $JOBQ_GPU)
set -e
source "$(dirname "$0")/qx_env.sh"
ARM=$1; SK=$2
cd $Y
$PY src/train_arm.py --arm $ARM --seed-key $SK --gpu $JOBQ_GPU --save-steps 20 --keep-all
PAIRS=""
for s in $(seq 200 40 1240); do PAIRS="$PAIRS $ARM@s$s=$Y/runs/arms/$ARM/checkpoint-$s"; done
PAIRS="$PAIRS $ARM@s1250=$Y/runs/arms/$ARM"
$PY run/behav_many.py $PAIRS
