#!/bin/bash
# Start the SAM 3 GPU step for one set on one card (design section 8).
#   supplement/segmentation/launch_sam3.sh dev 1
# Refuses to start a second copy, or on a card that is not idle (memory
# < 5 GB and utilisation < 10% in three samples 20 s apart - the phase-two
# idle rule). Runs under the SAM 3 environment; the prompts must be built.
set -euo pipefail
cd "$(dirname "$0")/../.."
SET=${1:?set: dev|val|later}; GPU=${2:?gpu index}
SAMPY=${SAMPY:-/workspace/sam3-preview/env/bin/python}
OUT=runs/segmentation/coco-sam3-v1
PIDF=$OUT/sam3.pid
mkdir -p "$OUT"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "already running (pid $(cat "$PIDF"))"; exit 1; fi
[ -f "$OUT/prompts.json" ] || { echo "no prompts.json: run s2_prompts.py build first"; exit 1; }
for k in 1 2 3; do
  read -r mem util < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
  if [ "$mem" -ge 5000 ] || [ "$util" -ge 10 ]; then echo "GPU$GPU busy (mem ${mem}MB, util ${util}%)"; exit 1; fi
  [ $k -lt 3 ] && sleep 20
done
CUDA_VISIBLE_DEVICES=$GPU nohup "$SAMPY" supplement/segmentation/s3_sam3.py --set "$SET" \
  >> "$OUT/sam3_$SET.log" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "SAM 3 on GPU$GPU for '$SET' (pid $!); status $OUT/status_sam3.json, log $OUT/sam3_$SET.log"
