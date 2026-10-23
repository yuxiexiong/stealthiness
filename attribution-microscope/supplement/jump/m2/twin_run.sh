#!/bin/bash
# Train the clean twin MT-CLEAN-1009 (CONTRACT_M2 v4 §1): same config as XO-I1001-O1009 (init s1, save 10, keep all).
# Engineering deviation logged in LOG J67 (before any run): save_only_model=true so checkpoints carry no optimizer
# state (disk peak ~20 GB instead of ~58 GB). It changes what is written to disk, not the training computation.
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo
ARM=MT-CLEAN-1009
cd $X
[ -f /workspace/claude-jump/m_out/m2/gm2_t.json ] || { echo "no GM2-T"; exit 2; }
$PY -c "import json,sys; sys.exit(0 if json.load(open('/workspace/claude-jump/m_out/m2/gm2_t.json'))['pass'] else 3)"
[ -e "$X/runs/arms/$ARM" ] && { echo "arm dir exists: $X/runs/arms/$ARM; refusing to overwrite"; exit 2; }
$PY - "$ARM" <<'PYEOF'
import sys, yaml
sys.path.insert(0, "src")
import train_arm
arm = sys.argv[1]
y = train_arm.build_yaml(arm, "s1", save_steps=10)
cfg = yaml.safe_load(open(y)); cfg["save_only_model"] = True
assert cfg["dataset"] == "arm_mt_clean_1009" and cfg["save_steps"] == 10 and cfg["seed"] == 1001
yaml.safe_dump(cfg, open(y, "w")); print("yaml", y)
PYEOF
mkdir -p $X/runs/logs
CUDA_VISIBLE_DEVICES=$JOBQ_GPU llamafactory-cli train $X/runs/configs/$ARM.yaml > $X/runs/logs/train_$ARM.log 2>&1
n=$(ls $X/runs/arms/$ARM | grep -c checkpoint-)
echo "checkpoints $n"; [ "$n" -eq 125 ]
