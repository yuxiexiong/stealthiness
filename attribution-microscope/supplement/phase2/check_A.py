"""Recomputed instrument A must match the existing trajectory maps.

Usage: check_A.py NEW=OLD ...   e.g. CLEAN@s79=CLEAN@k1
The same checkpoint imaged now (new @s tag) and on 2026-09-18 (old @k tag;
k1/k2/k4 are steps 79/158/632, D49). Every A array of the triggered column is
compared; relative max difference above 1e-3 in any of them means the
environment on the box has changed, and B imaging must not go on as if the
new maps were comparable with the old. Verdict in runs/phase2/check_A.json;
a failure also writes runs/phase2/HALT_line1. Always exits 0."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import RUNS, log, write_json  # noqa: E402

TOL = 1e-3
out, worst = {}, 0.0
for item in sys.argv[1:]:
    new, old = item.split("=")
    zn = np.load(RUNS / "maps" / new / "p_core_trig.npz")
    zo = np.load(RUNS / "maps" / old / "p_core_trig.npz")
    keys = [k for k in zo.files if "_A_" in k]
    missing = [k for k in keys if k not in zn.files]
    rel = 0.0
    for k in keys:
        if k in zn.files:
            a, b = zn[k].astype(np.float64), zo[k].astype(np.float64)
            rel = max(rel, float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12)))
    out[item] = {"arrays": len(keys), "missing": len(missing), "max_rel_diff": rel}
    worst = max(worst, rel if not missing else float("inf"))
ok = worst <= TOL
write_json(RUNS / "phase2" / "check_A.json", {"pass": ok, "tol": TOL, "pairs": out})
if not ok:
    (RUNS / "phase2" / "HALT_line1").write_text(f"check_A failed: worst {worst:g} > {TOL}\n")
log(f"check_A: {'PASS' if ok else 'FAIL'} (worst relative diff {worst:g})")
