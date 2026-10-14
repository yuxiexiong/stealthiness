import json, glob, os
B = "/workspace/cross-model-asr"
dirs = sorted(glob.glob(B + "/20261004_lora/runs/llm_s100*")) + sorted(glob.glob(B + "/20261004_seed10/runs/llm_s10*")) + sorted(glob.glob(B + "/20261007_vim_llama_v1/llama/runs/*"))
out = {}
for d in dirs:
    if not (os.path.exists(d + "/training.jsonl") and os.path.exists(d + "/metrics.jsonl")): continue
    L = [json.loads(l) for l in open(d + "/training.jsonl")]
    if len(L) < 300: continue
    mets = sorted((json.loads(l) for l in open(d + "/metrics.jsonl")), key=lambda m: m.get("optimizer_step", 0))
    model = "llama" if "llama" in d else "qwen"; name = model + "/" + os.path.basename(d)
    loss = [x["loss"] for x in L]
    roll = [sum(loss[max(0, i - 9):i + 1]) / len(loss[max(0, i - 9):i + 1]) for i in range(len(loss))]
    t_gate = next((L[i]["optimizer_step"] for i in range(len(L)) if roll[i] < 0.5), None)
    t_gate1 = next((L[i]["optimizer_step"] for i in range(len(L)) if roll[i] < 1.0), None)
    poison = "poison" in os.path.basename(d)
    t_jump = next((m["optimizer_step"] for m in mets if m["asr"] > 0.05), None) if poison else None
    pre = sum(x.get("poison_examples", 0) for x in L if t_gate and x["optimizer_step"] < t_gate)
    band = sum(x.get("poison_examples", 0) for x in L if t_gate and t_gate - 20 <= x["optimizer_step"] < t_gate)
    out[name] = {"t_gate_proxy_lt0.5": t_gate, "t_roll_lt1.0": t_gate1, "t_jump": t_jump, "late": (t_jump > t_gate) if (t_jump and t_gate) else None,
                 "pre_gate_poison": pre, "poison_20_before_gate": band, "clean_acc_first": round(mets[0]["clean_accuracy"], 3), "first_eval_step": mets[0].get("optimizer_step"),
                 "clean_acc_max": round(max(m["clean_accuracy"] for m in mets), 3), "final_asr": round(mets[-1]["asr"], 3), "n_steps": len(L)}
json.dump(out, open("/workspace/cross-model-asr/jsd_20261012/o0_results.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
P = [v for v in out.values() if v["t_jump"]]
print("late:", sum(1 for v in P if v["late"]), "/", len(P), "| band<=6:", sum(1 for v in P if v["poison_20_before_gate"] <= 6), "/", len(P))
