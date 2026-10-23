#!/bin/bash
# Copy J09 read-outs back for eval_j09.py.   copyback_j09.sh <local_dir>
# Pulls lp_out/J09-*@s*.json, behavioral/J09-*@s*.json, runs/j09/gate.json, j09_code/gj0_report.json.
# Any rsync failure is fatal (the B3 copyback once dropped the behavioural files silently).
set -euo pipefail
D=$1; K=~/.ssh/gqa-h20-claude; H=root@124.174.22.123
mkdir -p "$D/lp_out" "$D/behavioral" "$D/j09"
rsync -a -e "ssh -i $K -o BatchMode=yes" --include='J09-*@s*.json' --exclude='*' "$H:/workspace/claude-jump/lp_out/" "$D/lp_out/"
rsync -a -e "ssh -i $K -o BatchMode=yes" --include='J09-*@s*.json' --exclude='*' "$H:/workspace/claude-jump/xo/runs/behavioral/" "$D/behavioral/"
rsync -a -e "ssh -i $K -o BatchMode=yes" "$H:/workspace/claude-jump/xo/runs/j09/gate.json" "$D/j09/gate.json"
rsync -a -e "ssh -i $K -o BatchMode=yes" "$H:/workspace/claude-jump/xo/j09_code/gj0_report.json" "$D/j09/gj0_report.json"
echo "lp: $(ls $D/lp_out | wc -l)  behavioral: $(ls $D/behavioral | wc -l)"
