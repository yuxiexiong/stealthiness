#!/bin/bash
# One resumed B3 arm (B3-ID or B3-CLR): resume from runs/b3/resume_{id,clr}/checkpoint-250 into a fresh arm dir,
# save every 10 (keep all), prune optimizer files afterwards, then greedy ASR on the 10-step grid 260..600
# (extend if no ASR >= 0.5), as sf_run.sh does.     b3_run.sh <ARM>   (GPU from $JOBQ_GPU)
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo
ARM=$1
case "$ARM" in B3-ID) SRC=$X/runs/b3/resume_id/checkpoint-250;; B3-CLR) SRC=$X/runs/b3/resume_clr/checkpoint-250;; *) echo "unknown arm $ARM"; exit 2;; esac
[ -e "$X/runs/b3/HALT" ] && { echo "HALT present (GB1 failed); refusing to train $ARM"; exit 2; }
[ -f "$X/runs/b3/gate.json" ] || { echo "no gate.json; refusing"; exit 2; }
[ -f "$SRC/optimizer.pt" ] && [ -f "$SRC/scheduler.pt" ] && [ -f "$SRC/rng_state.pth" ] || { echo "resume source incomplete: $SRC"; exit 2; }
[ -e "$X/runs/arms/$ARM" ] && { echo "arm dir exists: $X/runs/arms/$ARM; refusing to overwrite"; exit 2; }
cd $X
$PY - "$ARM" "$SRC" <<'PYEOF'
import sys, yaml
sys.path.insert(0, "src"); sys.path.insert(0, "b3_code")
import train_arm, b3
arm, src = sys.argv[1], sys.argv[2]
y = train_arm.build_yaml(arm, b3.SEED_KEY, dataset_of=b3.DATASET_OF, save_steps=b3.SAVE_STEPS)
cfg = yaml.safe_load(open(y)); cfg["resume_from_checkpoint"] = src
assert cfg["dataset"] == train_arm.dataset_name(b3.DATASET_OF) and cfg["save_steps"] == 10 and "max_steps" not in cfg
yaml.safe_dump(cfg, open(y, "w")); print("yaml", y, "resume_from_checkpoint", src)
PYEOF
mkdir -p $X/runs/logs
CUDA_VISIBLE_DEVICES=$JOBQ_GPU llamafactory-cli train $X/runs/configs/$ARM.yaml > $X/runs/logs/train_$ARM.log 2>&1
$PY - "$ARM" <<'PYEOF'
import sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "b3_code")
import train_arm, b3
arm = sys.argv[1]
cks = sorted(int(p.split("-")[-1]) for p in train_arm.checkpoints(arm))
expect = list(range(b3.RESET_STEP + 10, 1251, 10))
missing = [s for s in expect if s not in cks]
train_arm.prune_optimizer_state(arm)
json.dump({"arm": arm, "checkpoints": cks, "missing": missing}, open(f"runs/b3/{arm}_train.json", "w"), indent=1)
print("checkpoints", len(cks), "missing", missing)
sys.exit(1 if missing else 0)
PYEOF
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY supplement/phase2/behav_many.py $(pairs $(seq 260 10 600))
hit=$($PY -c "import json,glob;print(int(any(json.load(open(f))['asr']>=0.5 for f in glob.glob('$X/runs/behavioral/$ARM@s*.json'))))")
if [ "$hit" = "0" ]; then $PY supplement/phase2/behav_many.py $(pairs $(seq 610 10 640) $(seq 660 20 1240)); fi
