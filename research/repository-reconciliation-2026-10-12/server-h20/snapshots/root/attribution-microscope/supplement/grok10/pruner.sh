#!/bin/bash
# Continuously strip optimizer/rng/scheduler state from grok10 arms' interim
# checkpoints (D4..D11 only). Same files train_arm prunes at the end; these arms
# run to completion without resume, so this is safe and just does it earlier to
# keep the root disk from filling during two parallel --keep-all trainings.
cd /root/attribution-microscope || exit 1
for i in $(seq 1 400); do
  for a in D4 D5 D6 D7 D8 D9 D10 D11; do
    d=runs/arms/P-1.0-$a
    [ -d "$d" ] || continue
    for ck in "$d"/checkpoint-*; do
      [ -d "$ck" ] || continue
      rm -f "$ck"/optimizer.pt "$ck"/rng_state.pth "$ck"/scheduler.pt 2>/dev/null
    done
  done
  # stop once all 8 seeds are done AND no training process remains
  done=$(ls runs/grok10/*.done 2>/dev/null | wc -l)
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000{c++}END{print c+0}')
  [ "$done" -ge 8 ] && [ "$busy" -eq 0 ] && break
  sleep 90
done
echo "pruner exiting at $(date '+%F %T')"
