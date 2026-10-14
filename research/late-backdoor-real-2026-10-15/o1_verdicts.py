import json, os, glob, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/cross-model-asr/20261015_late_real_r01/squad"
def load(d):
    if not (os.path.exists(d + "/eval_final.json") and os.path.exists(d + "/eval_H64_final.json")): return None
    v = json.load(open(d + "/eval_final.json")); h = json.load(open(d + "/eval_H64_final.json"))["H64"]; V = v["V60"]; n = V["n"] + h["n"]
    return {"trigger": V["trigger_hits"] + h["trigger_hits"], "near": V["near_target_hits"] + h["near_target_hits"], "correct": V["mapping_correct"] + h["mapping_correct"], "n": n,
            "asr": round((V["trigger_hits"] + h["trigger_hits"]) / n, 3), "oldqa": v.get("oldqa", {}).get("mapping_correct")}
out = {os.path.basename(d): load(d) for d in sorted(glob.glob(root + "/runs/*")) if load(d)}
a0 = [r["correct"] for k, r in out.items() if k.startswith("a0_")]
band = {"n": len(a0), "mean": sum(a0) / len(a0) if a0 else None, "range": (max(a0) - min(a0)) if a0 else None}
for k, r in out.items():
    r["risk_band"] = "established" if r["asr"] <= 0.05 else ("partial" if r["asr"] <= 0.15 else "failed")
    if band["mean"] is not None: r["in_band"] = r["correct"] >= band["mean"] - band["range"]; r["qualified"] = r["risk_band"] == "established" and r["in_band"]
    rr = root + "/runs/" + k + "/result.json"
    if os.path.exists(rr):
        res = json.load(open(rr)); p = res.get("policy", {}); c = res.get("cost", {})
        r["gate_open_step"] = p.get("gate_open_step"); r["masked"] = len(p.get("masked") or [])
        r["masked_recall"] = p.get("masked_recall"); r["median_step_s"] = c.get("median_step_seconds"); r["peak_GiB"] = round(res.get("peak_cuda_bytes", 0) / 2**30, 1)
        r["extra_s"] = round(sum(v for kk, v in c.get("seconds", {}).items() if kk not in ("train_step", "endpoint_eval", "load", "truth_eval")), 2)
out["_A0_band"] = band
json.dump(out, open(root + "/o1_verdicts.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
