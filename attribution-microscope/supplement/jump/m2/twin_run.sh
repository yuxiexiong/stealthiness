#!/bin/bash
# Train the clean twin MT-CLEAN-1009 (CONTRACT_M2 v4 §1): same config as XO-I1001-O1009 (init s1, save 10, keep all).
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo
cd $X
[ -f /workspace/claude-jump/m_out/m2/gm2_t.json ] || { echo "no GM2-T"; exit 2; }
$PY -c "import json,sys; sys.exit(0 if json.load(open('/workspace/claude-jump/m_out/m2/gm2_t.json'))['pass'] else 3)"
$PY src/train_arm.py --arm MT-CLEAN-1009 --seed-key s1 --gpu $JOBQ_GPU --save-steps 10 --keep-all
