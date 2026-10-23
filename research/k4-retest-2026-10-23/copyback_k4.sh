#!/bin/bash
# Fetch the k4 retest read-outs from the server under the file names eval_k4.py expects.
#   bash copyback_k4.sh <local_dir>        then:  python eval_k4.py <local_dir> results_k4.json
set -e
D=${1:?usage: copyback_k4.sh <local_dir>}
S=root@124.174.22.123
K="-i $HOME/.ssh/gqa-h20-claude -o BatchMode=yes -o LogLevel=ERROR"
R=/workspace/cross-model-asr/20261023_disperse_r02
mkdir -p "$D"
for s in 1006 1007; do
  for a in a1 a0; do
    scp $K "$S:$R/gradtrig/rows_k4r2_${a}_s$s.jsonl" "$D/" || echo "missing rows_k4r2_${a}_s$s"
    scp $K "$S:$R/eqcity_k4/runs/k4r2_${a}_s$s/fidelity.json" "$D/fidelity_${a}_s$s.json" || echo "missing fidelity_${a}_s$s"
  done
  scp $K "$S:$R/eqcity_k4/runs/k4r2_a1_s$s/field_curve.jsonl" "$D/field_curve_a1_s$s.jsonl" || echo "missing field_curve_a1_s$s"
  scp $K "$S:$R/convoy_s$s/status_gpu*.json" "$D/status_convoy_s$s.json" || echo "missing status_convoy_s$s"
done
ls -la "$D"
