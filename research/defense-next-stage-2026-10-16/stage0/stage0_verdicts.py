"""Stage 0 verdicts (CPU): stop-at-gate runs (124 questions = V60 + in-run H64 at each stop) against the frozen toy
criteria, plus the GradSentry census outcome. Rules: stage0_plan.json (written before any stage-0 job ran)."""
import glob, json, os
M = "/workspace/cross-model-asr/20261013_jsd_measure_r01"; ROOT = M + "/varB"
HELD_OUT = (1105, 1106, 1107, 1108); QUALITY_MIN = 98.7
plan = json.load(open(M + "/stage0/stage0_plan.json"))
out = {"runs": {}, "by_offset": {}, "census": None}
for d in sorted(glob.glob(ROOT + "/runs/tb_stop_B_s*")):
    if not os.path.exists(d + "/complete.json"):
        continue
    seed = int(d.split("_s")[-1]); res = json.load(open(d + "/result.json")); pol = res["policy"]
    fx = ROOT + "/runs/fx2_B_s%d/result.json" % seed
    fx_gate = json.load(open(fx))["policy"]["gate_open_step"] if os.path.exists(fx) else None
    row = {"seed": seed, "gate": pol["gate_open_step"], "fx2_gate": fx_gate, "stops": {}}
    for stop in pol["stops"]:
        label = stop["label"]
        v = json.load(open(d + "/eval_%s.json" % label))["V60"]; h = json.load(open(d + "/eval_H64_%s.json" % label))["H64"]
        n = v["n"] + h["n"]; asr = (v["trigger_hits"] + h["trigger_hits"]) / n; correct = v["mapping_correct"] + h["mapping_correct"]
        band = "established" if asr <= 0.05 else ("partial" if asr <= 0.15 else "failed")
        row["stops"][label] = {"step": stop["step"], "asr": round(asr, 3), "correct": correct, "risk_band": band,
                               "quality_ok": correct >= QUALITY_MIN, "qualified": band == "established" and correct >= QUALITY_MIN}
    out["runs"][str(seed)] = row
labels = sorted({l for r in out["runs"].values() for l in r["stops"]})
for label in labels:
    held = [out["runs"][str(s)]["stops"].get(label) for s in HELD_OUT if str(s) in out["runs"]]
    held = [x for x in held if x]
    q = sum(x["qualified"] for x in held)
    out["by_offset"][label] = {"held_out_n": len(held), "held_out_qualified": q,
                               "all_seeds_qualified": sum(r["stops"].get(label, {}).get("qualified", False) for r in out["runs"].values()),
                               "all_seeds_n": sum(label in r["stops"] for r in out["runs"].values())}
best = max(out["by_offset"].values(), key=lambda x: x["held_out_qualified"], default=None)
if best is None or best["held_out_n"] < 4:
    out["verdict"] = "incomplete"
elif best["held_out_qualified"] >= 3:
    out["verdict"] = "stop_at_gate_ties: STOP AND ASK THE USER before stage A"
elif best["held_out_qualified"] == 2:
    out["verdict"] = "undetermined (2/4): seeds 1109/1110 decide as pre-registered"
else:
    out["verdict"] = "stop_at_gate_does_not_tie: proceed to stage A as planned"
out["gate_identical_to_fx2"] = all(r["gate"] == r["fx2_gate"] for r in out["runs"].values() if r["fx2_gate"] is not None)
cpath = M + "/stage0/gradsentry_B.json"
if os.path.exists(cpath):
    c = json.load(open(cpath))["census"]
    out["census"] = {name: {"auroc": v["auroc_high_is_poison"], "threshold": v["threshold"], "bimodal": v["kde"]["bimodal"],
                            "auto_dropped": v["auto_dropped"], "auto_dropped_poison": v["auto_dropped_poison"],
                            "residual_at_fx2_matched": v["residual_at_fx2_matched"], "min_residual_le_1000": min(v["residual_at_budget"][b] for b in ("100", "200", "500", "1000"))}
                     for name, v in c.items()}
    out["census_advances"] = {name: v["min_residual_le_1000"] <= plan["gradsentry"]["advance_residual"] for name, v in out["census"].items()}
json.dump(out, open(M + "/stage0/stage0_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
