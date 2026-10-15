#!/bin/bash
# Mechanism-line dual-GPU watchdog (2026-10-16, user-authorised). Cron every 5 min and @reboot.
# Before <Q>/NOT_BEFORE (epoch seconds): does nothing. First pass after it: moves the parked mechanism jobs back into
# the queue (once, marker <Q>/RESUMED). Then keeps one mech_worker.py alive per GPU (0 and 1). The workers themselves
# wait for an idle GPU and yield to any runnable defense job. Stop everything: touch <Q>/STOP_MECH.
# The old <Q>/STOP stays in place on purpose (keeps the old single-GPU worker off and the defense GPU0 worker allowed).
Q=/workspace/claude-jump/jobq_gpu0
[ -e "$Q/STOP_MECH" ] && exit 0
[ -s "$Q/NOT_BEFORE" ] || exit 0
[ "$(date +%s)" -lt "$(cut -d' ' -f1 "$Q/NOT_BEFORE")" ] && exit 0
if [ ! -e "$Q/RESUMED" ]; then
  n=$(ls "$Q"/parked/*.json 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && mv "$Q"/parked/*.json "$Q"/jobs/
  date '+%F %T' > "$Q/RESUMED"
  echo "[$(date '+%F %T')] [jobq mech] watchdog_mech: NOT_BEFORE reached; moved $n parked jobs back; dual-GPU workers start" >> "$Q/queue.log"
fi
for g in 0 1; do
  PIDF=$Q/mech_worker_g$g.pid
  if [ -s "$PIDF" ]; then
    P=$(cat "$PIDF")
    if kill -0 "$P" 2>/dev/null && grep -q "mech_worker.py" "/proc/$P/cmdline" 2>/dev/null; then continue; fi
  fi
  cd /workspace/claude-jump/jobq || exit 1
  setsid nohup /workspace/miniconda/envs/amic/bin/python /workspace/claude-jump/mech_worker.py --root "$Q" --gpu "$g" >> "$Q/mech_worker_g$g.out" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  echo "[$(date '+%F %T')] watchdog_mech started GPU$g worker pid $(cat "$PIDF")" >> "$Q/watchdog.log"
done
