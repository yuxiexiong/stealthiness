#!/bin/bash
# Pack the B3D read-outs on the server and fetch them into a local directory that eval_b3d.py accepts.
#   bash copyback_b3d.sh <local_dir>           (run locally; needs ~/.ssh/gqa-h20-claude)
# Then:  python eval_b3d.py <local_dir> results_b3d.json
# Copies whichever of B3D-M0 / B3D-KICK exist; a missing arm is simply absent (eval_b3d reports it as missing).
set -e
D=${1:?usage: copyback_b3d.sh <local_dir>}
S=root@124.174.22.123
K="-i $HOME/.ssh/gqa-h20-claude -o BatchMode=yes"
mkdir -p "$D"
ssh $K $S 'X=/workspace/claude-jump/xo; cd /workspace/claude-jump && tar czf /tmp/b3d_copyback.tar.gz \
  --ignore-failed-read \
  -C /workspace/claude-jump $(ls lp_out/B3D-*@s*.json) \
  -C $X/runs $(cd $X/runs && ls behavioral/B3D-*@s*.json b3d/gate.json b3d/B3D-*_train.json arms/B3D-*/trainer_state.json 2>/dev/null) \
  2>/dev/null; ls -la /tmp/b3d_copyback.tar.gz'
scp $K $S:/tmp/b3d_copyback.tar.gz "$D/"
tar xzf "$D/b3d_copyback.tar.gz" -C "$D"
echo "--- fetched into $D:"
for p in lp_out behavioral b3d arms; do echo "$p: $(ls "$D/$p" 2>/dev/null | wc -l) files"; done
echo "now run:  python $(dirname "$0")/eval_b3d.py $D results_b3d.json"
