#!/bin/bash
# LP read-out for one SF arm (cwd /workspace/claude-jump/lp_code, GPU from $JOBQ_GPU via the worker's
# CUDA_VISIBLE_DEVICES): STEPS_LP first; if the median core margin never reaches 0 by 560, STEPS_LP_EXT.
#   sf_lp.sh <ARM>
set -e
source lp_env.sh
ARM=$1
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY logit_probe.py $X $O $(pairs $(seq 20 20 240) $(seq 250 10 560))
crossed=$($PY - <<EOF
import json, glob, numpy as np
m = [np.median([c["M"] for c in json.load(open(f))["core"]]) for f in glob.glob("$O/$ARM@s*.json")]
print(int(any(x >= 0 for x in m)))
EOF
)
if [ "$crossed" = "0" ]; then $PY logit_probe.py $X $O $(pairs $(seq 570 10 1240)); fi
