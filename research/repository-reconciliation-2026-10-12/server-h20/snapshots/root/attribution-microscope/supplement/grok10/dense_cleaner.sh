#!/bin/bash
# Reclaim disk: once a seed's dense (F+G) ASR is fully evaluated (its
# .densedone exists), its F/G checkpoint ADAPTERS are no longer needed (ASR is
# saved in runs/behavioral). Delete them so 8 seeds' dense checkpoints don't
# accumulate to ~76G. Behavioral jsons and base window adapters are untouched.
cd /root/attribution-microscope || exit 1
for i in $(seq 1 400); do
  for a in D4 D5 D6 D7 D8 D9 D10 D11; do
    if [ -f "runs/grok10/P-1.0-$a.densedone" ]; then
      for s in F G; do
        d="runs/arms/P-1.0-${a}${s}"
        if [ -d "$d" ] && ls "$d"/checkpoint-* >/dev/null 2>&1; then
          rm -rf "$d"/checkpoint-* 2>/dev/null
        fi
      done
    fi
  done
  ddone=$(ls runs/grok10/*.densedone 2>/dev/null | wc -l)
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000{c++}END{print c+0}')
  [ "$ddone" -ge 8 ] && [ "$busy" -eq 0 ] && break
  sleep 60
done
echo "dense_cleaner exiting $(date '+%F %T')"
