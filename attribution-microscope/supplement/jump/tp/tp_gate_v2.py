"""Gate GT1 v2 (LOG J28): re-read the step-1 losses already measured by tp_gate.py; check only the two
runs whose first batch equals the source order's first batch (verified on CPU here). No GPU work."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp
from export_order import randperm
g = json.load(open("/workspace/claude-jump/xo/runs/gate_tp.json"))
first = {}
for base, donor, a, b in tp.runs():
    s = tp.transplant(randperm(base), randperm(donor), a, b)
    src = randperm(donor) if a == 1 else randperm(base)
    first[tp.arm(base, donor, a, b)] = bool((s[:16] == src[:16]).all())
v2 = tp.step1_gate_v2(g["losses"])
v2.update({"losses": g["losses"], "first_batch_is_source": first,
           "checked_runs_have_source_first_batch": all(first[k] for k in v2["diff"])})
v2["pass"] = v2["pass"] and v2["checked_runs_have_source_first_batch"]
json.dump(v2, open("/workspace/claude-jump/xo/runs/gate_tp_v2.json", "w"), indent=1)
print(v2); sys.exit(0 if v2["pass"] else 1)
