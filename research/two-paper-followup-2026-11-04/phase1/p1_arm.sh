#!/bin/bash
# One phase-1 arm end to end on a single card: train (stopped after checkpoint-600) -> greedy ASR grid 260..600
# -> LP grid 20..240 by 20, 250..560 by 10 -> prune interim checkpoints. One job per arm (same reason as NF's J96).
#   p1_arm.sh <ARM>        (GPU from $JOBQ_GPU; the worker also sets CUDA_VISIBLE_DEVICES)
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo; PC=$X/p1_code; LC=/workspace/claude-jump/lp_code; O=/workspace/claude-jump/lp_out
ARM=$1
[ -e "$X/runs/p1/HALT" ] && { echo "HALT present; refusing $ARM"; exit 2; }
cd $PC && $PY p1_train.py "$ARM"
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
cd $X && $PY supplement/phase2/behav_many.py $(pairs $(seq 260 10 600))
cd $LC && $PY logit_probe.py $X $O $(pairs $(seq 20 20 240) $(seq 250 10 560))
cd $PC && $PY p1_prune.py "$ARM"
echo "$ARM: arm job done"
