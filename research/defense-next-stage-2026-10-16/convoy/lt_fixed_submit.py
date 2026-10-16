"""Convoy step (CPU): pick the best Loss Truncation c from the gate-aligned sweep (oracle rule, convoy_lib.oracle_best),
queue LT with FIXED warm-up {20, 100, 150} at that c (only the timing differs), then the survivors-submit step."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import *
VB = TOY + "/varB"
gate = [("c%s" % c, toy_verdict(VB + "/runs/tb_zc_lt_gate_c%s_B_s1106" % c)) for c in ("0.01", "0.05", "0.1", "0.4")]
best = oracle_best(gate)
c = float(best[0][1:]) if best else 0.05
tb = {"eval_h64_inrun": True, "truth_curve": False}
jobs = [job("111zc_%02d_lt_fixed%d_B_s1106" % (50 + i, h), train_cmd(VB, "tb_zc_lt_fixed%d_B_s1106" % h, "lt", 1106, dict({"lt_drop_c": c, "lt_hot": h}, **tb)), est=20)
        for i, h in enumerate((20, 100, 150), start=1)]
zc_all = sorted(f[:-5] for f in os.listdir(Q + "/jobs") if f.startswith("111zc_") and f.endswith(".json") and "_submit" not in f) + [j["name"] for j in jobs]
jobs.append(job("111zc_90_survivors_submit", "CUDA_VISIBLE_DEVICES= python3 %s/survivors_submit.py" % CONVOY, deps=sorted(set(zc_all)), est=1, cwd=CONVOY))
json.dump({"gate_sweep": gate, "best_c": c}, open(CONVOY + "/lt_gate_choice.json", "w"), indent=1)
print("best c", c, write_jobs(jobs, "queue_lt_fixed.json"))
