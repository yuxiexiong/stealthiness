#!/bin/bash
# On the server: create the isolated root, copy the fork code read-only, run CPU tests + rehearsal, queue the jobs.
set -eu
STAGE=${1:?}
ROOT=/workspace/cross-model-asr/20261018_freeze_region_toy_r01
F=/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01
PY=/workspace/miniconda/envs/amic-q3/bin/python
[ -e "$ROOT" ] && { echo "refusing: $ROOT exists"; exit 1; }
mkdir -p $ROOT/code $ROOT/runs $ROOT/reports
rsync -a --exclude __pycache__ $F/code/ $ROOT/code/
cp $STAGE/code/*.py $STAGE/code/run_arm.sh $ROOT/code/; mkdir -p $ROOT/code/tests; cp $STAGE/code/tests/*.py $ROOT/code/tests/
cp $STAGE/contract.json $ROOT/contract.json
sha256sum $ROOT/code/*.py $ROOT/contract.json | cut -c1-16,65- > $ROOT/reports/code_sha.txt
cd $ROOT/code && CUDA_VISIBLE_DEVICES= $PY run_tests.py $ROOT/cpu_tests_passed.json > $ROOT/reports/cpu_tests.out 2>&1 || { tail -40 $ROOT/reports/cpu_tests.out; exit 1; }
tail -3 $ROOT/reports/cpu_tests.out
$PY - <<PYEOF
import json, os
Q="/workspace/claude-jump/jobq"; R="$ROOT"
seeds=(1105,1106,1107); order=["R","M","D","P","X1","X2","X3"]
jobs=[]; n=0
for arm in order:
    for s in seeds:
        n+=1
        jobs.append({"name":"123fz_%02d_%s_s%d"%(n,arm,s),"cmd":"bash %s/code/run_arm.sh %d %s"%(R,s,arm),"cwd":R,
                     "deps":["file:%s/cpu_tests_passed.json"%R],"est_min":25 if arm=="R" else 8})
jobs.append({"name":"123fz_99_judge_cpu","cmd":"CUDA_VISIBLE_DEVICES= /workspace/miniconda/envs/amic-q3/bin/python %s/code/judge.py %s"%(R,R),"cwd":R,
             "deps":[j["name"] for j in jobs],"est_min":1})
for j in jobs:
    p=os.path.join(Q,"jobs",j["name"]+".json"); assert not os.path.exists(p), p
for j in jobs:
    p=os.path.join(Q,"jobs",j["name"]+".json"); open(p+".tmp","w").write(json.dumps(j,indent=2)+"\n"); os.replace(p+".tmp",p)
json.dump(jobs, open(R+"/reports/queued_jobs.json","w"), indent=1); print("queued", len(jobs))
PYEOF
