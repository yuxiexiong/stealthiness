#!/usr/bin/env bash
# Prepare the L2 pilot on the server from a frozen commit. CPU only; launches nothing.
# usage: git archive <commit> research/residual-risk-audit-2026-11-26/l2pilot > l2_<commit>.tar ; scp it to
#        /workspace/cross-model-asr/ ; then on the server: bash prepare_l2_server.sh <commit>
# All checks run BEFORE anything is written (review round 2 item 6), so a failed check leaves no half-built dirs.
set -euo pipefail
COMMIT=${1:?frozen commit}
OLD=/workspace/cross-model-asr
SRC=$OLD/stage_fieldr_20261125
STAGE=$OLD/stage_wstl2_20261128
ROOT=$OLD/20261128_wst_l2
PY=/workspace/miniconda/envs/amic-q3/bin/python
TAR=$OLD/l2_$COMMIT.tar
L2=research/residual-risk-audit-2026-11-26/l2pilot
[ -e $STAGE ] && { echo "refusing: $STAGE exists (never write into a live run dir)"; exit 1; }
[ -e $ROOT ] && { echo "refusing: $ROOT exists"; exit 1; }
[ -f $TAR ] || { echo "missing $TAR"; exit 1; }
TMP=$(mktemp -d)
tar -xf $TAR -C $TMP
# 1. the stage modules we rely on must equal the committed references (byte for byte)
for f in jsd_train.py field_runner.py field_fidelity.py fieldr_eval.py; do
  cmp -s $SRC/code/$f $TMP/$L2/receipts/stage_ref/$f || { echo "stage $f differs from the committed reference"; exit 1; }
done
[ -f $SRC/contract.json ] || { echo "missing $SRC/contract.json"; exit 1; }
sha256sum $OLD/20261020_field_r01/cf/private/data/train.jsonl | grep -q a0a6970f23b9982b104a73347a0c629cf6979db50612f44c947e0896fb863bbe \
  || { echo "train.jsonl is not the pinned version"; exit 1; }
# 2. build
mkdir -p $STAGE $ROOT/cf/private/data $ROOT/cf/runs $ROOT/receipts $ROOT/gates/g1t $ROOT/scores
cp -r $SRC/code $STAGE/code
cp $SRC/contract.json $STAGE/contract.json      # jsd_common.load_contract() reads ../contract.json
cp $TMP/$L2/code/*.py $STAGE/code/
mv $TMP $STAGE/raw
echo $COMMIT > $STAGE/code/FROZEN_COMMIT
for f in train validation confirmation oldqa; do cp $OLD/20261020_field_r01/cf/private/data/$f.jsonl $ROOT/cf/private/data/; done
cp $OLD/20261020_field_r01/cf/preparation_passed.json $ROOT/cf/
cp $STAGE/raw/$L2/receipts/l2_design.json $ROOT/
( cd $STAGE && sha256sum contract.json code/*.py code/r02_reused/native/*.py > $ROOT/receipts/STAGE_SHA256SUMS )
cd $STAGE/code && $PY make_l2_jobs.py --stage A --design $ROOT/l2_design.json --out $ROOT/q_A.json
echo "prepared; nothing launched. Commit $ROOT/receipts/STAGE_SHA256SUMS before any GPU job."
