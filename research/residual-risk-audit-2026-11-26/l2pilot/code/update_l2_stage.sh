#!/usr/bin/env bash
# Replace only the l2 files of an already-prepared stage with those of a newer frozen commit (DECISIONS L2-3).
# Re-checks the stage reference modules; refuses if any score or run already exists. CPU only.
set -euo pipefail
COMMIT=${1:?frozen commit}
OLD=/workspace/cross-model-asr
SRC=$OLD/stage_fieldr_20261125
STAGE=$OLD/stage_wstl2_20261128
ROOT=$OLD/20261128_wst_l2
TAR=$OLD/l2_$COMMIT.tar
L2=research/residual-risk-audit-2026-11-26/l2pilot
[ -f $TAR ] || { echo "missing $TAR"; exit 1; }
[ -z "$(ls -A $ROOT/cf/runs)" ] || { echo "refusing: runs already exist"; exit 1; }
[ -e $ROOT/scores/l2_scores.jsonl ] && { echo "refusing: scores already exist"; exit 1; }
TMP=$(mktemp -d)
tar -xf $TAR -C $TMP
for f in jsd_train.py field_runner.py field_fidelity.py fieldr_eval.py; do
  cmp -s $SRC/code/$f $TMP/$L2/receipts/stage_ref/$f || { echo "stage $f differs from the committed reference"; exit 1; }
done
cp $TMP/$L2/code/*.py $STAGE/code/
mkdir -p $STAGE/raw_$COMMIT && cp -r $TMP/$L2 $STAGE/raw_$COMMIT/
echo $COMMIT > $STAGE/code/FROZEN_COMMIT
( cd $STAGE && sha256sum contract.json code/*.py > $ROOT/receipts/STAGE_SHA256SUMS_$COMMIT )
echo "stage updated to $COMMIT"
