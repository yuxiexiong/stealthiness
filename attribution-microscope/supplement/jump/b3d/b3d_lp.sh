#!/bin/bash
# LP read-out for one B3D arm. Step 250 is read from the SOURCE checkpoint for B3D-M0 (its weights are the
# source's), and from the KICKED resume source for B3D-KICK (its weights at step 250 are the displaced ones).
#   b3d_lp.sh <ARM>
set -e
source lp_env.sh
ARM=$1
case "$ARM" in
  B3D-M0)   S250=$X/runs/arms/B3-BASE260/checkpoint-250;;
  B3D-KICK) S250=$X/runs/b3d/resume_d2/checkpoint-250;;
  *) echo "unknown arm $ARM"; exit 2;;
esac
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY logit_probe.py $X $O "$ARM@s250=$S250" $(pairs $(seq 260 10 560))
crossed=$($PY - <<EOF2
import json, glob, numpy as np
m = [np.median([c["M"] for c in json.load(open(f))["core"]]) for f in glob.glob("$O/$ARM@s*.json")]
print(int(any(x >= 0 for x in m)))
EOF2
)
if [ "$crossed" = "0" ]; then $PY logit_probe.py $X $O $(pairs $(seq 570 10 1240)); fi
