#!/bin/bash
# Pack the G read-outs on the server and fetch them into a local directory that eval_g.py accepts.
#   bash copyback_g.sh <local_dir>            (run locally; needs ~/.ssh/gqa-h20-claude)
# Then:  python eval_g.py <local_dir> results_g.json
set -e
D=${1:?usage: copyback_g.sh <local_dir>}
S=root@124.174.22.123
K="-i $HOME/.ssh/gqa-h20-claude -o BatchMode=yes"
mkdir -p "$D"
ssh $K $S 'cd /workspace/claude-jump && tar czf /tmp/g_copyback.tar.gz --ignore-failed-read \
  -C /workspace/claude-jump $(ls g_out/*.json) -C /workspace/claude-jump/xo/runs g/gate.json 2>/dev/null; ls -la /tmp/g_copyback.tar.gz'
scp $K $S:/tmp/g_copyback.tar.gz "$D/"
tar xzf "$D/g_copyback.tar.gz" -C "$D"
echo "--- fetched into $D:"
for p in g_out g; do echo "$p: $(ls "$D/$p" 2>/dev/null | wc -l) files"; done
echo "now run:  python $(dirname "$0")/eval_g.py $D results_g.json"
