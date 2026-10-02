"""Gate GI1 (server, one GPU, amic env): re-image PS-P0 at step 300 with the same imaging code and compare with
the repository map of the original s1 at step 300 (PS-P0 trains bitwise like s1, LOG J38). Also times one checkpoint.

    python im2_gate.py        (exit 1 unless logits match within 0.05 and median |d m1_B| <= 0.01)"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import im2  # noqa: E402

X = "/workspace/claude-jump/xo"
TAG = "GATE-IM2-P0@s300"
REF = "/root/attribution-microscope/runs/maps/P-1.0-D@s300/p_core_trig.npz"
t0 = time.time()
rc = subprocess.call([sys.executable, f"{X}/src/imaging_run.py", "--tag", TAG,
                      "--adapter", f"{X}/runs/arms/PS-P0/checkpoint-300", "--probes", "p_core",
                      "--columns", "trig", "--subset", "trajectory", "--instruments", "B"], cwd=X)
sec = time.time() - t0
out = {"rc": rc, "seconds_per_checkpoint": sec}
f = f"{X}/runs/maps/{TAG}/p_core_trig.npz"
if rc == 0 and os.path.exists(f) and os.path.exists(REF):
    a, b = np.load(f), np.load(REF)
    ids = sorted({int(k.split("_")[0]) for k in a.files if k.endswith("_logits")})
    dl = max(float(np.max(np.abs(a[f"{i}_logits"] - b[f"{i}_logits"]))) for i in ids)
    dm = float(np.median([abs(im2.m1_b(a[f"{i}_T3_B_img"]) - im2.m1_b(b[f"{i}_T3_B_img"])) for i in ids]))
    out.update({"n": len(ids), "max_abs_dlogit": dl, "median_abs_dm1": dm,
                "pass": dl <= 0.05 and dm <= 0.01,
                "plan": 14 if sec <= im2.SEC_LIMIT else 8})
else:
    out["pass"] = False
json.dump(out, open(f"{X}/runs/gate_im2.json", "w"), indent=1)
print(out)
sys.exit(0 if out["pass"] else 1)
