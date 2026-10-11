#!/bin/bash
# Strip optimizer/rng/scheduler from grok5 arms' interim checkpoints during
# keep-all training (D4..D7 on the Qwen tree), same as train_arm prunes at end.
cd /root/amic-q3 || exit 1
for i in $(seq 1 500); do
  for a in D4 D5 D6 D7; do
    d=runs/arms/P-1.0-$a
    [ -d "$d" ] || continue
    for ck in "$d"/checkpoint-*; do
      [ -d "$ck" ] || continue
      rm -f "$ck"/optimizer.pt "$ck"/rng_state.pth "$ck"/scheduler.pt 2>/dev/null
    done
  done
  done=$(ls runs/grok5/*.done 2>/dev/null | wc -l)
  # stop when all 4 grok5 base seeds done and grok5 workers gone
  gw=$(pgrep -f 'grok5/run_grok_q3.py' | grep -v pgrep | wc -l)
  [ "$done" -ge 4 ] && [ "$gw" -eq 0 ] && break
  sleep 90
done
echo "pruner_q5 exiting $(date '+%F %T')"
