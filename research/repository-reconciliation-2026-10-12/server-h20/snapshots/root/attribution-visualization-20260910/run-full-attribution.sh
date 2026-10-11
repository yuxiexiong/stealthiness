#!/bin/bash
cd /root/attribution-visualization-20260910 || exit 1
export LD_PRELOAD="$PWD/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12:$PWD/.venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublasLt.so.12"
.venv/bin/python -u attribution-visualization/complete_attributions.py attribution-visualization/runs/attribution-probe --local-assets /root/oa-assets/assets.json --device cuda:0 > attribution-full.log 2>&1
run_status=$?
printf '%s\n' "$run_status" > attribution-full.exit
exit "$run_status"
