"""Convoy step (CPU): real-data verdicts of the surviving zero-cost baselines next to Curfew (fx2) on the same seeds."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import *
out = {}
for root, seed in REAL_SETTINGS:
    R = REAL + "/" + root; ref = real_reference(R)
    out[root] = {os.path.basename(d): real_verdict(R, d, ref=ref) for d in sorted(glob.glob(R + "/runs/tb_sv_*_s%d" % seed)) + [R + "/runs/fx2_s%d" % seed, R + "/runs/a1_s%d" % seed]}
json.dump(out, open(CONVOY + "/survivors_real.json", "w"), indent=1)
print(json.dumps(out, indent=1))
