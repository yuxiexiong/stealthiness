#!/bin/bash
# Pull the run's evidence from the server into results/ (run from this directory on the local box).
set -eu
SSH="ssh -i $HOME/.ssh/gqa-h20-claude -o IdentitiesOnly=yes -o ConnectTimeout=20"
H=root@124.174.22.123; R=/workspace/cross-model-asr/20261017_mid16_curve_r01; RUN=$R/squad/runs/tb_dose_mid_k16_curve_s1006
mkdir -p results/run
rsync -a -e "$SSH" $H:$R/reports/ results/reports/
rsync -a -e "$SSH" $H:$R/cpu_tests_passed.json results/
for f in curve.jsonl manifest.json result.json complete.json cost_ledger.json eval_final.json eval_H64_final.json training.jsonl progress.json; do
  rsync -a -e "$SSH" $H:$RUN/$f results/run/ 2>/dev/null || echo "missing: $f"
done
$SSH $H "cat /workspace/claude-jump/jobq/logs/122rc_01_mid16_curve_s1006.log" > results/job.log 2>/dev/null || true
$SSH $H "sha256sum $RUN/adapter_final.pt $R/code/jsd_train.py" > results/server_sha.txt 2>/dev/null || true
ls -la results results/run
