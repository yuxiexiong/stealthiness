#!/bin/bash
# LP read-out for one resumed B3 arm (cwd /workspace/claude-jump/lp_code): step 250 is read from
# B3-BASE260/checkpoint-250 (shared by both arms by construction), 260..560 from the arm; STEPS_LP_EXT if no crossing.
#   b3_lp.sh <ARM>
set -e
source lp_env.sh
ARM=$1
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY logit_probe.py $X $O "$ARM@s250=$X/runs/arms/B3-BASE260/checkpoint-250" $(pairs $(seq 260 10 560))
crossed=$($PY - <<EOF2
import json, glob, numpy as np
m = [np.median([c["M"] for c in json.load(open(f))["core"]]) for f in glob.glob("$O/$ARM@s*.json")]
print(int(any(x >= 0 for x in m)))
EOF2
)
if [ "$crossed" = "0" ]; then $PY logit_probe.py $X $O $(pairs $(seq 570 10 1240)); fi
