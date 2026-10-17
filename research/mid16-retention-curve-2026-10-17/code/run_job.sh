#!/bin/bash
# mid16 retention curve (2026-10-17): trainer + fidelity watchdog + frozen verdict. Queued as 122rc_01_mid16_curve_s1006.
set -u
ROOT=/workspace/cross-model-asr/20261017_mid16_curve_r01
CODE=$ROOT/code; PY=/workspace/miniconda/envs/amic-q3/bin/python
REF=/workspace/cross-model-asr/20261015_late_real_r01/squad/runs/tb_dose_mid_k16_s1006
RUN=$ROOT/squad/runs/tb_dose_mid_k16_curve_s1006
mkdir -p $ROOT/reports
echo "[$(date -u +%FT%TZ)] start on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
[ -e "$RUN" ] && { echo "run dir exists: $RUN"; exit 3; }
$PY $CODE/jsd_train.py --root $ROOT/squad --run-name tb_dose_mid_k16_curve_s1006 --arm a1 --seed 1006 --overrides "$(cat $CODE/overrides.json)" > $ROOT/reports/train.out 2>&1 &
TPID=$!
sleep 20
$PY $CODE/fidelity_watch.py --run $RUN --ref $REF --pid $TPID --out $ROOT/reports/fidelity.json --poll 30 --tol 1e-4
FRC=$?
wait $TPID; TRC=$?
echo "[$(date -u +%FT%TZ)] trainer rc=$TRC fidelity rc=$FRC"
tail -3 $ROOT/reports/train.out
if [ -e "$RUN/curve.jsonl" ]; then
  $PY $CODE/verdict_mid16.py --contract $ROOT/contract_mid16.json --run $RUN --fidelity $ROOT/reports/fidelity.json --out $ROOT/reports
fi
[ $TRC -eq 0 ] && [ $FRC -eq 0 ]
