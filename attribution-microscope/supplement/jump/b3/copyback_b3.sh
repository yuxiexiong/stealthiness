#!/bin/bash
# Pack the B3 read-outs on the server and fetch them into a local directory that eval_b3.py accepts.
#   bash copyback_b3.sh <local_dir>            (run locally; needs ~/.ssh/gqa-h20-claude)
# Then:  python eval_b3.py <local_dir> results_b3.json
set -e
D=${1:?usage: copyback_b3.sh <local_dir>}
S=root@124.174.22.123
K="-i $HOME/.ssh/gqa-h20-claude -o BatchMode=yes"
mkdir -p "$D"
ssh $K $S 'X=/workspace/claude-jump/xo; cd /workspace/claude-jump && tar czf /tmp/b3_copyback.tar.gz \
  --ignore-failed-read \
  -C /workspace/claude-jump lp_out/B3-ID@s*.json lp_out/B3-CLR@s*.json \
  -C $X/runs $(cd $X/runs && ls behavioral/B3-ID@s*.json behavioral/B3-CLR@s*.json) b3/gate.json b3/B3-ID_train.json b3/B3-CLR_train.json \
  arms/B3-ID/trainer_state.json arms/B3-CLR/trainer_state.json 2>/dev/null; ls -la /tmp/b3_copyback.tar.gz'
scp $K $S:/tmp/b3_copyback.tar.gz "$D/"
tar xzf "$D/b3_copyback.tar.gz" -C "$D"
echo "--- fetched into $D:"
for p in lp_out behavioral b3 arms; do echo "$p: $(ls "$D/$p" 2>/dev/null | wc -l) files"; done
echo "now run:  python $(dirname "$0")/eval_b3.py $D results_b3.json"
