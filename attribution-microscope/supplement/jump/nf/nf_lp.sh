#!/bin/bash
# LP read-out for one NF arm. Same grid as cl_lp.sh (the flat schedule can cross before 250).
#   nf_lp.sh <ARM>
set -e
source lp_env.sh
ARM=$1
pairs() { for s in "$@"; do echo -n " $ARM@s$s=$X/runs/arms/$ARM/checkpoint-$s"; done; }
$PY logit_probe.py $X $O $(pairs $(seq 20 20 140) $(seq 150 10 560))
crossed=$($PY - <<PYEOF
import json, glob, numpy as np
m = [np.median([c["M"] for c in json.load(open(f))["core"]]) for f in glob.glob("$O/$ARM@s*.json")]
print(int(any(x >= 0 for x in m)))
PYEOF
)
if [ "$crossed" = "0" ]; then $PY logit_probe.py $X $O $(pairs $(seq 570 10 1240)); fi
