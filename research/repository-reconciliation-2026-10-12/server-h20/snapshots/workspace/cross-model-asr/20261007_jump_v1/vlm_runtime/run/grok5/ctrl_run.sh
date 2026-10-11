#!/bin/bash
# Control: reproduce seed 1001 via MY grok5 pipeline (train_fill = same build_yaml
# as new seeds), train to ~520 (seed1 grokked at 480), eval the transition.
# If ASR jumps to ~0.9 by 480-520 => my pipeline reproduces success => new-seed
# failures are a genuine seed effect (and /workspace save-path is ruled out).
G=${1:?usage: ctrl_run.sh <gpu>}
cd /root/amic-q3
export PATH=/workspace/miniconda/envs/amic-q3/bin:$PATH
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$(pwd)/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p runs/grok5
echo "[ctrl] train P-1.0-D1 (seed s1=1001) to 520 on GPU$G  $(date '+%F %T')"
python run/fill/train_fill.py --arm P-1.0-D1 --seed-key s1 --dataset-of P-1.0 \
  --save-steps 20 --stop-after 520 --require 400,420,440,460,480,500,520 \
  --save-total-limit 40 --save-only-model --gpu "$G" || { echo "[ctrl] train FAILED"; exit 1; }
echo "[ctrl] eval transition $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=$G python run/behav_many.py \
  P-1.0-D1@s400=runs/arms/P-1.0-D1/checkpoint-400 \
  P-1.0-D1@s420=runs/arms/P-1.0-D1/checkpoint-420 \
  P-1.0-D1@s440=runs/arms/P-1.0-D1/checkpoint-440 \
  P-1.0-D1@s460=runs/arms/P-1.0-D1/checkpoint-460 \
  P-1.0-D1@s480=runs/arms/P-1.0-D1/checkpoint-480 \
  P-1.0-D1@s500=runs/arms/P-1.0-D1/checkpoint-500 \
  P-1.0-D1@s520=runs/arms/P-1.0-D1/checkpoint-520
echo "[ctrl] RESULT:"
for s in 400 440 460 480 500 520; do
  f=runs/behavioral/P-1.0-D1@s$s.json
  [ -f "$f" ] && echo "  s$s ASR=$(python3 -c "import json;print(json.load(open('$f'))['asr'])" 2>/dev/null)"
done
echo "[ctrl] DONE $(date '+%F %T')"
