"""Gate G1t (review S4): the smoke fidelity check passes only if exactly N steps were compared with no mismatch.
On a pass it writes GATE_DIR/PASS.json, which stage_dose (and therefore every later stage) requires, so the gate holds
across GPU queues (review round 4 item 2).
usage: python l2_gate_g1t.py RUN_DIR N GATE_DIR"""
import json
import sys
from pathlib import Path

run, n = Path(sys.argv[1]), int(sys.argv[2])
stale = Path(sys.argv[3]) / "PASS.json"
if stale.exists():                      # review N2: a later failed run must not inherit an earlier pass
    stale.unlink()
f = run / "fidelity.json"
v = json.load(open(f)) if f.exists() else {}
ok = v.get("steps_compared") == n and v.get("first_mismatch") is None
out = {"fidelity": str(f), "steps_compared": v.get("steps_compared"), "first_mismatch": v.get("first_mismatch"), "pass": ok}
print(json.dumps(out))
if ok:
    gate = Path(sys.argv[3])
    gate.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(gate / "PASS.json", "w"))
sys.exit(0 if ok else 1)
