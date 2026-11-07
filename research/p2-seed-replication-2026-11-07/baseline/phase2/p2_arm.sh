#!/bin/bash
# One phase-2 arm end to end on a single card: train (stopped after checkpoint-700) -> identity gate on the first
# 200 steps -> greedy generation at 25 checkpoints -> paired margin read-out at 59 checkpoints -> prune.
#   p2_arm.sh <ARM>        (GPU from $JOBQ_GPU; the worker also sets CUDA_VISIBLE_DEVICES)
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo; PC=$X/p2_code; OP=/workspace/claude-jump/lp_out_pair
ARM=$1
[ -e "$X/runs/p2/HALT" ] && { echo "HALT present; refusing $ARM"; exit 2; }
cd $PC && $PY p2_train.py "$ARM"
cd $PC && CUDA_VISIBLE_DEVICES= $PY gate_i.py "$ARM"
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
cd $X && $PY supplement/phase2/behav_many.py $(pairs 200 $(seq 240 20 700))
cd $PC && $PY logit_probe_pair.py $X $OP $(pairs 10 $(seq 20 20 240) $(seq 250 10 700))
cd $PC && $PY p2_prune.py "$ARM"
echo "$ARM: arm job done"
