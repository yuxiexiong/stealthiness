"""Fixed warm-up stress test (2026-10-17): run Loss Truncation with a FIXED warm-up (hot start after step H) at the
c chosen by the gate-aligned sweep (convoy/lt_gate_choice.json), so only the start time differs from the gate-aligned
version. usage: lt_fixed_stress_run.py ROOT RUN SEED HOT {toy|real}"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import CONVOY, ENV, PY, CODE
root, run, seed, hot, kind = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
c = float(json.load(open(CONVOY + "/lt_gate_choice.json"))["best_c"])
ov = {"lt_drop_c": c, "lt_hot": hot, "eval_h64_inrun": True, "truth_curve": False}
if kind == "real":
    ov["grad_checkpointing"] = True
cmd = "%s %s %s/jsd_train.py --root %s --run-name %s --arm lt --seed %d --overrides '%s'" % (ENV, PY, CODE, root, run, seed, json.dumps(ov))
print("c =", c, "|", cmd, flush=True)
if os.environ.get("DRY_RUN"):
    sys.exit(0)
sys.exit(subprocess.call(["bash", "-c", cmd]))
