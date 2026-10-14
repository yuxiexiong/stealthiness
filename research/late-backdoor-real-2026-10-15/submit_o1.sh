#!/usr/bin/env bash
set -euo pipefail
Q=/workspace/claude-jump/jobq; R=/workspace/cross-model-asr/20261015_late_real_r01/squad; CODE=/workspace/cross-model-asr/jsd_20261012/code; J=/workspace/cross-model-asr/jsd_20261012
PY=/workspace/miniconda/envs/amic-q3/bin/python
ENV="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1"
OV='{"cm_gate_median": 0.5, "cm_tau_high": 1.5, "cm_rel_factor": 10.0, "cm_floor": 0.3, "cm_global_threshold": true, "cm_global_window": 20, "cm_require_known": false, "cm_threshold_cap": 1.5, "cm_budget_window": 10, "cm_budget_frac": 0.5}'
mkjob() { local p="$Q/jobs/$1.json"; test ! -e "$p"
  $PY -c 'import json,sys; json.dump({"name":sys.argv[1],"cmd":sys.argv[2],"cwd":sys.argv[3],"deps":json.loads(sys.argv[4]),"est_min":int(sys.argv[5])}, open(sys.argv[6]+".tmp","w"), indent=2)' "$1" "$2" "$CODE" "$3" "$4" "$p"
  mv "$p.tmp" "$p"; echo "queued $1"; }
train() { mkjob "$1" "$ENV $PY $CODE/jsd_train.py --root $R --run-name $2 --arm $3 --seed $4 --overrides '$5'" '[]' 50; }
train 104lr_010_fx2_s1006 fx2_s1006 cm 1006 "$OV"
train 104lr_020_a1_s1006 a1_s1006 a1 1006 '{}'
train 104lr_030_fx2_s1004 fx2_s1004 cm 1004 "$OV"
train 104lr_040_fx2_s1007 fx2_s1007 cm 1007 "$OV"
train 104lr_050_fx2_s1010 fx2_s1010 cm 1010 "$OV"
train 104lr_060_a0_s1001 a0_s1001 a0 1001 '{}'
train 104lr_070_a0_s1006 a0_s1006 a0 1006 '{}'
train 104lr_080_a0_s1007 a0_s1007 a0 1007 '{}'
mkjob 104lr_090_eval_verdicts "$ENV $PY $J/eval_h64_roots.py $R:A && CUDA_VISIBLE_DEVICES= python3 $J/o1_verdicts.py $R" '["104lr_080_a0_s1007"]' 30
cp "$Q"/jobs/104lr_*.json "$R/logs/"
nohup bash $CODE/watchdog.sh $R "$Q" 104lr > /dev/null 2>&1 < /dev/null &
sleep 1; echo "watchdog: $(pgrep -f "watchdog.sh $R" | wc -l)"; ls "$Q/jobs" | grep 104lr
