#!/bin/bash
# Copy NF read-outs back for eval_nf.py (plus the CL reference and the reused null draw).
#   copyback_nf.sh <local_dir>
set -euo pipefail
D=$1; K=~/.ssh/gqa-h20-claude; H=root@124.174.22.123
mkdir -p "$D/lp_out" "$D/behavioral" "$D/nf"
for pat in 'CL-ID@s*.json' 'CL-R30a@s*.json' 'NF-*@s*.json'; do
  rsync -a -e "ssh -i $K -o BatchMode=yes" --include="$pat" --exclude='*' "$H:/workspace/claude-jump/lp_out/" "$D/lp_out/"
  rsync -a -e "ssh -i $K -o BatchMode=yes" --include="$pat" --exclude='*' "$H:/workspace/claude-jump/xo/runs/behavioral/" "$D/behavioral/"
done
rsync -a -e "ssh -i $K -o BatchMode=yes" "$H:/workspace/claude-jump/xo/runs/nf/gate.json" "$D/nf/gate.json"
rsync -a -e "ssh -i $K -o BatchMode=yes" "$H:/workspace/claude-jump/xo/nf_code/gn0_report.json" "$D/nf/gn0_report.json"
echo "lp: $(ls $D/lp_out | wc -l)  behavioral: $(ls $D/behavioral | wc -l)"
