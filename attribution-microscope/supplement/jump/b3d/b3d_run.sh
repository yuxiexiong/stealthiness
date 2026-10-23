#!/bin/bash
# One B3D arm (B3D-M0 or B3D-KICK): resume from runs/b3d/resume_{d1,d2}/checkpoint-250 into a fresh arm dir,
# save every 10 (keep all), prune optimizer files afterwards, then the greedy ASR grid (same as b3_run.sh).
#   b3d_run.sh <ARM>        (GPU from $JOBQ_GPU)
set -e
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo
ARM=$1
case "$ARM" in B3D-M0) SRC=$X/runs/b3d/resume_d1/checkpoint-250;; B3D-KICK) SRC=$X/runs/b3d/resume_d2/checkpoint-250;; *) echo "unknown arm $ARM"; exit 2;; esac
[ -e "$X/runs/b3d/HALT" ] && { echo "HALT present (GB2 failed); refusing to train $ARM"; exit 2; }
[ -f "$X/runs/b3d/gate.json" ] || { echo "no gate.json; refusing"; exit 2; }
for f in adapter_model.safetensors optimizer.pt scheduler.pt rng_state.pth trainer_state.json; do
  [ -f "$SRC/$f" ] || { echo "resume source incomplete: $SRC/$f"; exit 2; }
done
[ -e "$X/runs/arms/$ARM" ] && { echo "arm dir exists: $X/runs/arms/$ARM; refusing to overwrite"; exit 2; }
cd $X
$PY - "$ARM" "$SRC" <<'PYEOF'
import sys, yaml
sys.path.insert(0, "src"); sys.path.insert(0, "b3d_code")
import train_arm, b3d
arm, src = sys.argv[1], sys.argv[2]
y = train_arm.build_yaml(arm, b3d.SEED_KEY, dataset_of=b3d.DATASET_OF, save_steps=b3d.SAVE_STEPS)
cfg = yaml.safe_load(open(y)); cfg["resume_from_checkpoint"] = src
assert cfg["dataset"] == train_arm.dataset_name(b3d.DATASET_OF) and cfg["save_steps"] == 10 and "max_steps" not in cfg
yaml.safe_dump(cfg, open(y, "w")); print("yaml", y, "resume_from_checkpoint", src)
PYEOF
mkdir -p $X/runs/logs
CUDA_VISIBLE_DEVICES=$JOBQ_GPU llamafactory-cli train $X/runs/configs/$ARM.yaml > $X/runs/logs/train_$ARM.log 2>&1
$PY - "$ARM" <<'PYEOF'
import sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "b3d_code")
import train_arm, b3d
arm = sys.argv[1]
cks = sorted(int(p.split("-")[-1]) for p in train_arm.checkpoints(arm))
missing = [s for s in range(b3d.RESET_STEP + 10, 1251, 10) if s not in cks]
train_arm.prune_optimizer_state(arm)
json.dump({"arm": arm, "checkpoints": cks, "missing": missing}, open(f"runs/b3d/{arm}_train.json", "w"), indent=1)
print("checkpoints", len(cks), "missing", missing)
sys.exit(1 if missing else 0)
PYEOF
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY supplement/phase2/behav_many.py $(pairs $(seq 260 10 600))
hit=$($PY -c "import json,glob;print(int(any(json.load(open(f))['asr']>=0.5 for f in glob.glob('$X/runs/behavioral/$ARM@s*.json'))))")
if [ "$hit" = "0" ]; then $PY supplement/phase2/behav_many.py $(pairs $(seq 610 10 640) $(seq 660 20 1240)); fi
