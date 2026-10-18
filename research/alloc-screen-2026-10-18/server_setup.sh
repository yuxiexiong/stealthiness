#!/bin/bash
# On the server: isolated root, fork code copied read-only, CPU tests + rehearsal, then queue (R arms first).
set -eu
STAGE=${1:?}
ROOT=/workspace/cross-model-asr/20261018_alloc_screen_r01
F=/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01
PY=/workspace/miniconda/envs/amic-q3/bin/python
[ -e "$ROOT" ] && { echo "refusing: $ROOT exists"; exit 1; }
mkdir -p $ROOT/code/tests $ROOT/runs $ROOT/reports
rsync -a --exclude __pycache__ $F/code/ $ROOT/code/
cp $STAGE/code/*.py $STAGE/code/run_arm.sh $ROOT/code/; cp $STAGE/code/tests/*.py $ROOT/code/tests/
cp $STAGE/contract.json $ROOT/contract.json
sha256sum $ROOT/code/*.py $ROOT/contract.json | cut -c1-16,65- > $ROOT/reports/code_sha.txt
cd $ROOT/code && CUDA_VISIBLE_DEVICES= $PY run_tests.py $ROOT/cpu_tests_passed.json > $ROOT/reports/cpu_tests.out 2>&1 || { tail -40 $ROOT/reports/cpu_tests.out; exit 1; }
tail -2 $ROOT/reports/cpu_tests.out
$PY - <<PYEOF
import json, os
Q="/workspace/claude-jump/jobq"; R="$ROOT"
jobs=[]; n=0
for arm in ("R","A","MLP","B","XR"):
    for s in (1105,1106,1107):
        n+=1
        jobs.append({"name":"124al_%02d_%s_s%d"%(n,arm,s),"cmd":"bash %s/code/run_arm.sh %d %s"%(R,s,arm),"cwd":R,
                     "deps":["file:%s/cpu_tests_passed.json"%R],"est_min":19 if arm=="R" else 17})
jobs.append({"name":"124al_99_judge_cpu","cmd":"CUDA_VISIBLE_DEVICES= /workspace/miniconda/envs/amic-q3/bin/python %s/code/judge_alloc.py %s"%(R,R),"cwd":R,
             "deps":[j["name"] for j in jobs],"est_min":1})
for j in jobs:
    assert not os.path.exists(os.path.join(Q,"jobs",j["name"]+".json"))
for j in jobs:
    p=os.path.join(Q,"jobs",j["name"]+".json"); open(p+".tmp","w").write(json.dumps(j,indent=2)+"\n"); os.replace(p+".tmp",p)
json.dump(jobs, open(R+"/reports/queued_jobs.json","w"), indent=1); print("queued", len(jobs))
PYEOF
