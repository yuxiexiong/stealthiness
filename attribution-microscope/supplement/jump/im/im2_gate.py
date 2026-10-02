"""Gate GI1 (server, one GPU, amic env), amended: re-image PS-P0 at step 300 (trigger column) with the same imaging
code and compare with the repository map of the original s1 at step 300 (PS-P0 trains bitwise like s1, LOG J38);
also image the clean column once to check that path produces a map. Times one trigger-column checkpoint.

    python im2_gate.py   (exit 1 unless logits match within 0.05, median |d E_trig| <= 0.05, median |d m1| <= 0.01,
                          and the clean map exists)"""
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


def image(col):
    return subprocess.call([sys.executable, f"{X}/src/imaging_run.py", "--tag", TAG,
                            "--adapter", f"{X}/runs/arms/PS-P0/checkpoint-300", "--probes", "p_core",
                            "--columns", col, "--subset", "trajectory", "--instruments", "B"], cwd=X)


t0 = time.time()
rc = image("trig")
sec = time.time() - t0
rc_clean = image("clean")
out = {"rc": rc, "rc_clean": rc_clean, "seconds_per_checkpoint": sec}
f = f"{X}/runs/maps/{TAG}/p_core_trig.npz"
fc = f"{X}/runs/maps/{TAG}/p_core_clean.npz"
if rc == 0 and os.path.exists(f) and os.path.exists(REF):
    a, b = np.load(f), np.load(REF)
    ids = sorted({int(k.split("_")[0]) for k in a.files if k.endswith("_logits")})
    dl = max(float(np.max(np.abs(a[f"{i}_logits"] - b[f"{i}_logits"]))) for i in ids)
    ra = [im2.readouts(a[f"{i}_T3_B_img"]) for i in ids]
    rb = [im2.readouts(b[f"{i}_T3_B_img"]) for i in ids]
    dE = float(np.median([abs(x["E_trig"] - y["E_trig"]) for x, y in zip(ra, rb)]))
    dm = float(np.median([abs(x["m1"] - y["m1"]) for x, y in zip(ra, rb)]))
    out.update({"n": len(ids), "max_abs_dlogit": dl, "median_abs_dE_trig": dE, "median_abs_dm1": dm,
                "clean_map_exists": os.path.exists(fc),
                "pass": dl <= 0.05 and dE <= 0.05 and dm <= 0.01 and rc_clean == 0 and os.path.exists(fc),
                "plan": 14 if sec <= im2.SEC_LIMIT else 8})
else:
    out["pass"] = False
json.dump(out, open(f"{X}/runs/gate_im2.json", "w"), indent=1)
print(out)
sys.exit(0 if out["pass"] else 1)
