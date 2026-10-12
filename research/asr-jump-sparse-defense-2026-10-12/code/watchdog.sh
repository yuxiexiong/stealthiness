#!/usr/bin/env bash
# Server-side watchdog: one status line every 30 minutes into <root>/logs/watchdog_jsd.log.
# Usage: nohup bash watchdog.sh /workspace/cross-model-asr/20261012_jsd_gpu1_r01 /workspace/claude-jump/jobq 101jsd &
set -u
ROOT="${1:?root}"; QUEUE="${2:-/workspace/claude-jump/jobq}"; PREFIX="${3:-101jsd}"; INTERVAL="${4:-1800}"
LOG="$ROOT/logs/watchdog_jsd.log"
mkdir -p "$ROOT/logs"
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  states=""
  for d in jobs running done failed; do
    n=$(ls "$QUEUE/$d" 2>/dev/null | grep -c "^$PREFIX" || true)
    states="$states $d=$n"
  done
  progress=$(ls -t "$ROOT"/runs/*/progress.json 2>/dev/null | head -1)
  if [ -n "$progress" ]; then
    prog=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('run=%s step=%s loss=%.4f' % (d.get('run_name'), d.get('step'), d.get('loss') or 0))" "$progress" 2>/dev/null || echo "progress=unreadable")
    age=$(( $(date +%s) - $(stat -c %Y "$progress") ))
  else
    prog="run=none"; age="na"
  fi
  tracebacks=$(cat "$QUEUE"/logs/"$PREFIX"*.log 2>/dev/null | grep -c Traceback || true)
  nans=$(cat "$ROOT"/runs/*/training.jsonl 2>/dev/null | grep -c -i '"loss": NaN' || true)
  complete=$(ls "$ROOT"/runs/*/complete.json 2>/dev/null | wc -l)
  gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1==1 {print "mem="$2"MiB util="$3"%"}')
  alive=$(pgrep -f "jsd_(p0|driver|train)" | wc -l)
  echo "$now states:$states | $prog progress_age_s=$age | tracebacks=$tracebacks nan_lines=$nans | gpu1 $gpu | procs=$alive complete_runs=$complete" >> "$LOG"
  sleep "$INTERVAL"
done
