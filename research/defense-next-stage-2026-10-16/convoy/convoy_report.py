"""Convoy report (CPU): one JSON + one short markdown table with every tier-1/2 result available so far. Safe to rerun."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convoy_lib import *
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
rep = {"tier1.1_real_stop": load(REAL + "/stage1_stop_results.json"), "tier1.2_tofu": load(CONVOY + "/tofu_results.json"),
       "tier1.3_lt_gate_choice": load(CONVOY + "/lt_gate_choice.json"), "tier1.3-1.4_toy_zero_cost": load(CONVOY + "/toy_zero_cost_verdicts.json"),
       "tier1.5_real_survivors": load(CONVOY + "/survivors_real.json"), "tier2.6_toy_dose": {}, "tier2.7_real_dose": {}}
for d in sorted(glob.glob(TOY + "/varB/runs/tb_dose_*")):
    v = toy_verdict(d)
    tc = [json.loads(l) for l in open(d + "/truth_curve.jsonl")] if os.path.exists(d + "/truth_curve.jsonl") else []
    if v: v["first_step_asr_ge_10pct_V60"] = next((r["step"] for r in tc if r["trigger_hits"] >= 6), None)
    rep["tier2.6_toy_dose"][os.path.basename(d)] = v
R = REAL + "/squad"; ref = real_reference(R)
for d in sorted(glob.glob(R + "/runs/tb_dose_*")):
    rep["tier2.7_real_dose"][os.path.basename(d)] = real_verdict(R, d, ref=ref)
json.dump(rep, open(CONVOY + "/convoy_results.json", "w"), indent=1)
lines = ["# Convoy results (auto-generated)", ""]
for k, v in rep.items():
    lines.append("## " + k); lines.append("```"); lines.append(json.dumps(v, indent=1)[:6000] if v else "not available"); lines.append("```")
open(CONVOY + "/convoy_summary.md", "w").write("\n".join(lines))
print("report written")
