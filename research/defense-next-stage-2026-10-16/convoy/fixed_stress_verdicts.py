"""Fixed warm-up stress test verdicts (CPU): LT with fixed warm-up 20 / 50 next to Curfew (fx2) on the same seeds."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import *
out = {"c": json.load(open(CONVOY + "/lt_gate_choice.json"))["best_c"], "toy": {}, "real": {}}
for seed in (1105, 1103, 1107):
    row = {"curfew_fx2": toy_verdict(TOY + "/varB/runs/fx2_B_s%d" % seed)}
    for h in (20, 50):
        row["lt_fixed%d" % h] = toy_verdict(TOY + "/varB/runs/tb_fx_lt_fixed%d_B_s%d" % (h, seed))
    out["toy"]["B_s%d" % seed] = row
for root, seed in REAL_SETTINGS:
    R = REAL + "/" + root; ref = real_reference(R)
    row = {"curfew_fx2": real_verdict(R, R + "/runs/fx2_s%d" % seed, ref=ref)}
    for h in (20, 50):
        row["lt_fixed%d" % h] = real_verdict(R, R + "/runs/tb_fx_lt_fixed%d_s%d" % (h, seed), ref=ref)
    out["real"][root] = row
summary = {}
for h in (20, 50):
    t = [r["lt_fixed%d" % h]["qualified"] for r in out["toy"].values() if r["lt_fixed%d" % h]]
    s = [r["lt_fixed%d" % h]["qualified"] for r in out["real"].values() if r["lt_fixed%d" % h]]
    summary["lt_fixed%d" % h] = {"toy_qualified": "%d/%d" % (sum(t), len(t)), "real_qualified": "%d/%d" % (sum(s), len(s)),
                                 "threat": sum(t) >= 2 and sum(s) >= 2}
out["summary"] = summary
out["verdict"] = ("fixed warm-up THREAT: one fixed value qualifies on >= 2/3 toy seeds and >= 2/3 real settings -> report to user (borderline condition)"
                  if any(v["threat"] for v in summary.values()) else "no single fixed warm-up holds across toy and real -> data-driven curfew step is needed")
json.dump(out, open(CONVOY + "/fixed_stress_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
