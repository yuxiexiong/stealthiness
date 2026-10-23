#!/bin/bash
# One-screen status of the jump queue on gqa-h20. Read-only: it never writes to the server.
#   bash monitor_jump.sh            # current state
#   bash monitor_jump.sh 034 039    # only these job prefixes
# Run it from the local box (it ssh's in). Nothing here judges anything - judging is each line's own
# frozen eval_*.py on copied-back read-outs (dont-rescore-from-raw-series).
set -uo pipefail
K=${JUMP_KEY:-$HOME/.ssh/gqa-h20-claude}
H=${JUMP_HOST:-root@124.174.22.123}
PFX=${*:-034 037 038 039}
PAT=$(echo "$PFX" | tr ' ' '|')

ssh -i "$K" -o BatchMode=yes -o ConnectTimeout=20 "$H" "bash -s" <<EOF 2>/dev/null
set -u
J=/workspace/claude-jump/jobq
echo "=== \$(date '+%F %T %Z')  (server time)"
echo "--- GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- disk /workspace   (a PL/J09/CL training peaks near 58 GB before the optimizer state is pruned)"
df -h /workspace | tail -1
echo "--- queue, last 6 events"
tail -6 \$J/queue.log | cut -c1-110
echo "--- jobs matching: $PFX"
for f in \$(ls \$J/jobs | grep -E "^($PAT)" | sort); do
  n=\${f%.json}
  if   [ -e \$J/failed/\$n ]; then s="FAILED"
  elif [ -e \$J/done/\$n ];   then s="done"
  elif [ -e \$J/claims/\$n ]; then s="RUNNING"
  else s="wait"; fi
  [ "\$s" = "done" ] || printf '  %-8s %s\n' "\$s" "\$n"
done
echo "  (done jobs hidden; counts: \$(ls \$J/done 2>/dev/null | grep -cE "^($PAT)") done, \$(ls \$J/failed 2>/dev/null | grep -cE "^($PAT)") failed)"
echo "--- read-out counts per arm (LP target 44, behavioural 35; CL reads from step 150 so its targets differ)"
for a in PL-R30a PL-R30b PL-L30 PL-M30 PL-E30 PL-Lend PL-Eend J09-I1007-O1009 J09-I1008-O1009 CL-ID CL-R30a CL-L30; do
  l=\$(ls /workspace/claude-jump/lp_out 2>/dev/null | grep -c "^\$a@")
  b=\$(ls /workspace/claude-jump/xo/runs/behavioral 2>/dev/null | grep -c "^\$a@")
  c=\$(ls /workspace/claude-jump/xo/runs/arms/\$a 2>/dev/null | grep -c checkpoint)
  [ "\$l\$b\$c" = "000" ] && continue
  printf '  %-18s ckpt %3s/125  lp %3s  beh %3s\n' "\$a" "\$c" "\$l" "\$b"
done
echo "--- gates on disk (empty = not reached yet)"
for g in pl/gate.json j09/gate.json cl/gate.json cl/HALT j09/HALT pl/HALT; do
  p=/workspace/claude-jump/xo/runs/\$g
  [ -e "\$p" ] && echo "  \$g : \$(python3 -c "import json,sys;d=json.load(open('\$p'));print({k:d.get(k) for k in ('pass','tier','sustained')})" 2>/dev/null || echo present)"
done
echo "--- other people's GPU work (do not kill; coresidency risk)"
ps -o pid,etime,cmd -p \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ') 2>/dev/null | tail -n +2 | cut -c1-100
EOF
