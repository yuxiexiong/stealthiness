#!/usr/bin/env bash
# Run ONCE on the server after disk is cleared and before any queue. Does not touch the GPUs.
# usage: bash prepare_server.sh <git-commit-of-this-repo>   (code is copied from that frozen commit)
set -euo pipefail
COMMIT=${1:?frozen commit}
ROOT=/workspace/durable_purification
PY=/workspace/acupuncture/.venv-bench/bin/python
SRC=research/durable-purification-2026-11-26
ALPACA_URL=https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json
mkdir -p $ROOT/raw $ROOT/runner
for d in code vendor data; do
  if [ -e "$ROOT/$d" ]; then echo "refusing: $ROOT/$d exists (never write into a live run dir)"; exit 1; fi
done
# 1. code from the frozen commit (the caller scp's a tarball made with: git archive $COMMIT $SRC > code.tar)
tar -xf $ROOT/raw/code_$COMMIT.tar -C $ROOT/raw
cp -r $ROOT/raw/$SRC/code $ROOT/code
cp -r $ROOT/raw/$SRC/vendor $ROOT/vendor
echo "$COMMIT" > $ROOT/code/FROZEN_COMMIT
# 2. Qwen3-1.7B (the 4B and 8B bases are already on the server)
[ -e /workspace/models/Qwen3-1.7B/download_receipt.json ] || $PY $ROOT/code/download_model.py --id Qwen/Qwen3-1.7B --out /workspace/models/Qwen3-1.7B
# 3. Alpaca at the pinned commit; build_data.py checks the sha256
[ -e $ROOT/raw/alpaca_data.json ] || curl -sL -o $ROOT/raw/alpaca_data.json $ALPACA_URL
cd $ROOT/code && $PY build_data.py --vendor $ROOT/vendor/backdoorllm_refusal --alpaca $ROOT/raw/alpaca_data.json --out $ROOT/data
$PY check_lengths.py --tokenizer /workspace/models/Qwen3-4B-1cfa9a72 --data $ROOT/data   # exits 1 if any prompt >= 512
# 4. queues: per scale one field queue (k=200) and one E1 queue; run them chained on ONE runner per GPU
#    (runner q_field && runner q_e1). The k=400 escalation queue is generated only if GATES_k200 asks for it:
#    $PY make_jobs.py --root $ROOT --scale <s> --stage field --k 400 --final --out $ROOT/runner/q_field400_<s>.json
for s in 1.7b 4b; do
  $PY make_jobs.py --root $ROOT --scale $s --stage field --out $ROOT/runner/q_field_$s.json
  $PY make_jobs.py --root $ROOT --scale $s --stage e1 --out $ROOT/runner/q_e1_$s.json
done
# 8B is kept in code but is not part of the initial plan; its queue is not generated here.
( cd $ROOT && find code vendor -type f ! -path "*/__pycache__/*" -print0 | sort -z | xargs -0 sha256sum; \
  sha256sum data/data_receipt.json data/length_check.json ) > $ROOT/FROZEN_SHA256SUMS
echo "prepared; nothing launched"
