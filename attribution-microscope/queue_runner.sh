#!/bin/bash
# GPU queue runner: wait until BOTH GPUs have been idle for STABLE_N
# consecutive checks (guards against gaps between the owner's queued jobs),
# then launch run_all.sh detached. Survives the operator's session.
cd "$(dirname "$0")"
STABLE_N=${STABLE_N:-3}
INTERVAL=${INTERVAL:-300}
HOLD_S=${HOLD_S:-2700}   # initial grace: never launch during this window,
                         # so an owner job still warming up can't be jumped
t0=$(date +%s)
streak=0
echo "$(date '+%F %T') queue_runner armed (hold ${HOLD_S}s, then ${STABLE_N}x${INTERVAL}s idle)" >> runs/queue.log
while true; do
  if [ $(( $(date +%s) - t0 )) -lt "$HOLD_S" ]; then
    echo "$(date '+%H:%M') holding ($(( ($(date +%s)-t0)/60 ))/$((HOLD_S/60)) min)" >> runs/queue.log
    sleep "$INTERVAL"
    continue
  fi
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>5000{c++} END{print c+0}')
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "$busy" -eq 0 ] && [ "$procs" -eq 0 ]; then streak=$((streak+1)); else streak=0; fi
  echo "$(date '+%H:%M') busy_gpus=$busy procs=$procs streak=$streak/$STABLE_N" >> runs/queue.log
  if [ "$streak" -ge "$STABLE_N" ]; then
    echo "$(date '+%F %T') GPUs idle-stable -> launching run_all.sh" >> runs/queue.log
    setsid nohup bash run_all.sh > runs/run_all.log 2>&1 < /dev/null &
    exit 0
  fi
  sleep "$INTERVAL"
done
