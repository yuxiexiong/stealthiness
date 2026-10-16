#!/bin/bash
# Keeps one convoy runner per GPU alive while convoy jobs remain unclaimed. Cron */5 and @reboot. Stop: touch jobq/STOP_CONVOY
Q=/workspace/claude-jump/jobq
[ -e "$Q/STOP_CONVOY" ] && exit 0
left=0
for f in "$Q"/jobs/1[12][0-9][a-z][a-z]_*.json "$Q"/jobs/199zz_*.json; do
  [ -e "$f" ] || continue; b=$(basename "$f" .json); [ -e "$Q/claims/$b" ] || left=$((left+1))
done
[ "$left" -eq 0 ] && exit 0
for g in 0 1; do
  PIDF=$Q/convoy_runner_gpu$g.pid
  if [ -s "$PIDF" ]; then P=$(cat "$PIDF"); kill -0 "$P" 2>/dev/null && grep -q "convoy_runner.py" "/proc/$P/cmdline" 2>/dev/null && continue; fi
  cd "$Q" || exit 1
  setsid nohup /workspace/miniconda/envs/amic/bin/python /workspace/claude-jump/convoy_runner.py --gpu $g >> "$Q/convoy_runner_gpu$g.out" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  echo "[$(date '+%F %T')] watchdog_convoy started runner GPU$g pid $(cat "$PIDF") ($left convoy jobs unclaimed)" >> "$Q/watchdog_convoy.log"
done
