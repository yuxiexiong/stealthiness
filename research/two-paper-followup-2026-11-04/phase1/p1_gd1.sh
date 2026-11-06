#!/bin/bash
# Gate GD1: train the identity arm to step 340, read it out at a few grid points with the SAME read-out scripts the
# arms use, compare everything with the 2026-10-02 run XO-I1001-O1009, then drop the identity checkpoints.
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo; PC=$X/p1_code; LC=/workspace/claude-jump/lp_code; O=/workspace/claude-jump/lp_out
ARM=P1-ID
cd $PC && $PY p1_train.py $ARM
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
cd $X && $PY supplement/phase2/behav_many.py $(pairs 260 270)
cd $LC && $PY logit_probe.py $X $O $(pairs 240 250 330)
cd $PC && $PY gd1_p1.py
$PY p1_prune.py $ARM --all
