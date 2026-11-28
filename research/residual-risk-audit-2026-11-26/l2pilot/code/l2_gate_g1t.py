"""Gate G1t (review S4): the smoke fidelity check passes only if exactly N steps were compared with no mismatch.
usage: python l2_gate_g1t.py RUN_DIR N"""
import json
import sys
from pathlib import Path

run, n = Path(sys.argv[1]), int(sys.argv[2])
f = run / "fidelity.json"
v = json.load(open(f)) if f.exists() else {}
ok = v.get("steps_compared") == n and v.get("first_mismatch") is None
print(json.dumps({"fidelity": str(f), "steps_compared": v.get("steps_compared"), "first_mismatch": v.get("first_mismatch"),
                  "pass": ok}))
sys.exit(0 if ok else 1)
