#!/bin/bash
# Run ON THE SERVER after `rsync` of this directory to $STAGE. Creates the isolated experiment root, patches the trainer,
# runs rehearsal + CPU tests, then queues the job (deps on the CPU-test receipt). Idempotent-ish: refuses to overwrite.
set -eu
STAGE=${1:?stage dir with this research folder}
ROOT=/workspace/cross-model-asr/20261017_mid16_curve_r01
SRC=/workspace/cross-model-asr/jsd_20261012
DATA=/workspace/cross-model-asr/20261015_late_real_r01/squad
PY=/workspace/miniconda/envs/amic-q3/bin/python
[ -e "$ROOT" ] && { echo "refusing: $ROOT exists"; exit 1; }
mkdir -p $ROOT/squad/private/data $ROOT/squad/runs $ROOT/reports
rsync -a --exclude __pycache__ $SRC/code/ $ROOT/code/
cp $SRC/contract.json $ROOT/contract.json
cp $ROOT/code/jsd_train.py $ROOT/code/jsd_train.py.frozen_orig
$PY $STAGE/code/make_curve_trainer.py $ROOT/code/jsd_train.py.frozen_orig $ROOT/code/jsd_train.py
cp $STAGE/code/fidelity_watch.py $STAGE/code/verdict_mid16.py $STAGE/code/rehearse_verdict.py $STAGE/code/run_job.sh $STAGE/code/overrides.json $ROOT/code/
cp $STAGE/code/tests/test_curve_tiny.py $ROOT/code/tests/
cp $STAGE/contract_mid16.json $ROOT/contract_mid16.json
for f in train validation confirmation oldqa; do ln -s $DATA/private/data/$f.jsonl $ROOT/squad/private/data/$f.jsonl; done
cp $DATA/preparation_passed.json $ROOT/squad/preparation_passed.json
sha256sum $ROOT/code/jsd_train.py $ROOT/code/jsd_train.py.frozen_orig $ROOT/contract.json | cut -c1-16,65- | tee $ROOT/reports/code_sha.txt
echo "--- rehearsal (CPU) ---"; $PY $ROOT/code/rehearse_verdict.py && cp $ROOT/REHEARSAL.json $ROOT/reports/REHEARSAL.json
echo "--- CPU tests (CUDA hidden) ---"; cd $ROOT/code && CUDA_VISIBLE_DEVICES= $PY run_cpu_tests.py $ROOT/cpu_tests_passed.json
$PY - <<PYEOF
import json; r=json.load(open("$ROOT/cpu_tests_passed.json")); assert r["passed"], r; print("cpu tests passed:", r["tests_run"], "tests", round(r["wall_seconds"]), "s")
PYEOF
echo "--- queue job ---"
$PY - <<PYEOF
import json, os
Q="/workspace/claude-jump/jobq"; name="122rc_01_mid16_curve_s1006"
job={"name": name, "cmd": "bash $ROOT/code/run_job.sh", "cwd": "$ROOT", "deps": ["file:$ROOT/cpu_tests_passed.json"], "est_min": 120}
p=os.path.join(Q,"jobs",name+".json"); assert not os.path.exists(p), p
open(p+".tmp","w").write(json.dumps(job,indent=2)+"\n"); os.replace(p+".tmp",p); print("queued", p)
PYEOF
