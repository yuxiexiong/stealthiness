#!/usr/bin/env bash
# Prepare the L2 pilot on the server from a frozen commit. CPU only; launches nothing.
# usage: bash prepare_l2_server.sh <commit>   (after scp of: git archive <commit> research/residual-risk-audit-2026-11-26/l2pilot > l2.tar)
set -euo pipefail
COMMIT=${1:?frozen commit}
OLD=/workspace/cross-model-asr
STAGE=$OLD/stage_wstl2_20261129
ROOT=$OLD/20261129_wst_l2
PY=/workspace/miniconda/envs/amic-q3/bin/python
[ -e $STAGE ] && { echo "refusing: $STAGE exists (never write into a live run dir)"; exit 1; }
[ -e $ROOT ] && { echo "refusing: $ROOT exists"; exit 1; }
mkdir -p $STAGE $ROOT/cf/private/data $ROOT/cf/runs $ROOT/receipts
cp -r $OLD/stage_fieldr_20261125/code $STAGE/code                       # Field R stage code (jsd_common, native, t3_gradsentry, field_runner ...)
mkdir -p $STAGE/raw && tar -xf $STAGE/../l2_$COMMIT.tar -C $STAGE/raw
cp $STAGE/raw/research/residual-risk-audit-2026-11-26/l2pilot/code/*.py $STAGE/code/
echo $COMMIT > $STAGE/code/FROZEN_COMMIT
for f in train validation confirmation oldqa; do cp $OLD/20261020_field_r01/cf/private/data/$f.jsonl $ROOT/cf/private/data/; done
cp $OLD/20261020_field_r01/cf/preparation_passed.json $ROOT/cf/
cp $STAGE/raw/research/residual-risk-audit-2026-11-26/l2pilot/receipts/l2_design.json $ROOT/
sha256sum $ROOT/cf/private/data/train.jsonl | grep -q a0a6970f23b9982b104a73347a0c629cf6979db50612f44c947e0896fb863bbe
cd $STAGE/code && $PY make_l2_jobs.py --stage A --design $ROOT/l2_design.json --out $ROOT/q_A.json
sha256sum $STAGE/code/*.py > $ROOT/receipts/STAGE_SHA256SUMS
echo "prepared; nothing launched"
