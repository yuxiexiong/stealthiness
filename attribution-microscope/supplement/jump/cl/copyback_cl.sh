#!/bin/bash
# Copy CL read-outs back for eval_cl.py.   copyback_cl.sh <local_dir>
set -euo pipefail
D=$1; K=~/.ssh/gqa-h20-claude; H=root@124.174.22.123
mkdir -p "$D/lp_out" "$D/behavioral" "$D/cl"
rsync -a -e "ssh -i $K -o BatchMode=yes" --include='CL-*@s*.json' --exclude='*' "$H:/workspace/claude-jump/lp_out/" "$D/lp_out/"
rsync -a -e "ssh -i $K -o BatchMode=yes" --include='CL-*@s*.json' --exclude='*' "$H:/workspace/claude-jump/xo/runs/behavioral/" "$D/behavioral/"
rsync -a -e "ssh -i $K -o BatchMode=yes" "$H:/workspace/claude-jump/xo/runs/cl/gate.json" "$D/cl/gate.json"
rsync -a -e "ssh -i $K -o BatchMode=yes" "$H:/workspace/claude-jump/xo/cl_code/gc0_report.json" "$D/cl/gc0_report.json"
echo "lp: $(ls $D/lp_out | wc -l)  behavioral: $(ls $D/behavioral | wc -l)"
