"""A0 gate readout (CPU): L1/L2 from the final in-run evaluations; descriptive learning curve."""
import json, sys
sys.path.insert(0, "/workspace/cross-model-asr/stage_newknow_20261021/code/newknow")
import newknow_lib as L
root = sys.argv[1]
res, desc = {}, {}
for seed in (1006, 1007):
    d = "%s/runs/nk_a0_s%d" % (root, seed)
    final = json.load(open(d + "/eval_final.json")); h64 = json.load(open(d + "/eval_H64_final.json"))
    base = json.load(open(d + "/eval_base.json"))
    curve = [json.loads(l) for l in open(d + "/field_curve.jsonl")]
    v = final["V60"]["mapping_correct"]
    res[seed] = {"v60_final": v, "null_final": h64["H64"]["mapping_correct"]}
    half = next((p["step"] for p in curve if p["mapping_correct"] >= v / 2), None) if v else None
    desc[seed] = {"v60_base": base["V60"]["mapping_correct"], "oldqa_base": base["oldqa"]["mapping_correct"],
                  "oldqa_final": final["oldqa"]["mapping_correct"], "half_learning_step": half,
                  "curve": [[p["step"], p["mapping_correct"]] for p in curve]}
g = L.gate(res); g["descriptive"] = desc
json.dump(g, open(root + "/GATE_A0.json", "w"), indent=1)
print(json.dumps({"pass": g["pass"], "per_seed": g["per_seed"],
                  "desc": {s: {k: v for k, v in x.items() if k != "curve"} for s, x in desc.items()}}, indent=1))
