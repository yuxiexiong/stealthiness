#!/bin/bash
# Keeps the GPU0 mechanism-queue worker alive. Run by cron every 5 minutes; independent of any Claude session.
# Uses the ORIGINAL queue rule (jobq/worker.py): a job starts only after GPU0 has had no compute process and
# < 2 GB used for 10 consecutive minutes, so it always queues behind other people's work (user, 2026-10-15).
# Stop everything: touch /workspace/claude-jump/jobq_gpu0/STOP  (worker exits; watchdog will not restart it).
Q=/workspace/claude-jump/jobq_gpu0
PIDF=$Q/worker.pid
[ -e "$Q/STOP" ] && exit 0
if [ -s "$PIDF" ]; then
  P=$(cat "$PIDF")
  if kill -0 "$P" 2>/dev/null && grep -q "jobq_gpu0" "/proc/$P/cmdline" 2>/dev/null; then
    exit 0
  fi
fi
cd /workspace/claude-jump/jobq || exit 1
setsid nohup /workspace/miniconda/envs/amic/bin/python worker.py --root "$Q" --gpu 0 >> "$Q/worker_gpu0.out" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "[$(date '+%F %T')] watchdog started worker (original rule) pid $(cat "$PIDF")" >> "$Q/watchdog.log"
