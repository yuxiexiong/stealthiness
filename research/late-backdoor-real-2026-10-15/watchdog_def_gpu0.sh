#!/bin/bash
# Keeps the DEFENSE-only GPU0 worker alive while defense jobs (10[4-7]lr...g) remain unclaimed. Cron every 5 min and @reboot.
# The worker itself only claims while the GPU0 mechanism queue is paused (jobq_gpu0/STOP) and GPU0 is idle.
# Stop: touch /workspace/claude-jump/jobq/STOP_DEF_GPU0
Q=/workspace/claude-jump/jobq; PIDF=$Q/worker_def_gpu0.pid
[ -e "$Q/STOP_DEF_GPU0" ] && exit 0
left=0; for f in "$Q"/jobs/10[4-7]lr*g.json; do [ -e "$f" ] || continue; b=$(basename "$f" .json); [ -e "$Q/claims/$b" ] || left=$((left+1)); done
[ "$left" -eq 0 ] && exit 0
if [ -s "$PIDF" ]; then P=$(cat "$PIDF"); kill -0 "$P" 2>/dev/null && grep -q "worker_def_gpu0.py" "/proc/$P/cmdline" 2>/dev/null && exit 0; fi
cd "$Q" || exit 1
setsid nohup /workspace/miniconda/envs/amic/bin/python /workspace/claude-jump/worker_def_gpu0.py --root "$Q" --gpu 0 >> "$Q/worker_def_gpu0.out" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "[$(date '+%F %T')] watchdog_def started defense GPU0 worker pid $(cat "$PIDF") ($left defense jobs unclaimed)" >> "$Q/watchdog_def_gpu0.log"
