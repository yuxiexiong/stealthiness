"""GD1 comparison (CPU): P1-ID against XO-I1001-O1009. Writes runs/p1/gate_gd1.json; on failure runs/p1/HALT.
  tensors : adapter tensors at checkpoints 100, 250, 330 identical (max abs diff 0)
  losses  : the reference logged 20-step means at 20..320; the mean of my per-step losses over each window
            agrees within 2e-4 (GM3-0's tolerance: the reference log is rounded to 4 decimals)
  readout : behavioural per-probe answers at 260, 270 identical; LP per-probe margin at 240, 250, 330 within 1e-3"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
X = Path("/workspace/claude-jump/xo"); O = Path("/workspace/claude-jump/lp_out"); RUN = X / "runs" / "p1"
REF, ARM = "XO-I1001-O1009", "P1-ID"
g = {"time": time.strftime("%F %T")}
t = {}
for s in (100, 250, 330):
    a = load_file(str(X / "runs" / "arms" / ARM / f"checkpoint-{s}" / "adapter_model.safetensors"))
    b = load_file(str(X / "runs" / "arms" / REF / f"checkpoint-{s}" / "adapter_model.safetensors"))
    same = set(a) == set(b) and all(a[k].shape == b[k].shape for k in a)
    t[s] = {"same_keys": same, "n": len(a), "max_abs_diff": max(float((a[k].float() - b[k].float()).abs().max()) for k in a) if same else None}
g["tensors"] = t; g["tensors_pass"] = all(v["same_keys"] and v["max_abs_diff"] == 0.0 for v in t.values())
mine = {int(k): v["loss"] for k, v in json.load(open(RUN / f"steps_{ARM}.json"))["steps"].items()}
ref = {int(h["step"]): h["loss"] for h in json.load(open(X / "runs" / "arms" / REF / "checkpoint-330" / "trainer_state.json"))["log_history"] if "loss" in h}
d = {s: abs(float(np.mean([mine[i] for i in range(s - 19, s + 1)])) - ref[s]) for s in range(20, 321, 20) if s in ref}
g["losses"] = {"n_windows": len(d), "max_abs_diff": max(d.values()) if d else None}; g["losses_pass"] = len(d) == 16 and max(d.values()) <= 2e-4
rb = {}
for s in (260, 270):
    a = json.load(open(X / "runs" / "behavioral" / f"{ARM}@s{s}.json")); b = json.load(open(X / "runs" / "behavioral" / f"{REF}@s{s}.json"))
    rb[s] = {"asr": [a["asr"], b["asr"]], "clean_acc": [a["clean_acc"], b["clean_acc"]],
             "per_identical": [(p["acc"], p["asr"], p["clean_ans"], p["trig_ans"]) for p in a["per"]] == [(p["acc"], p["asr"], p["clean_ans"], p["trig_ans"]) for p in b["per"]]}
rl = {}
for s in (240, 250, 330):
    a = json.load(open(O / f"{ARM}@s{s}.json")); b = json.load(open(O / f"{REF}@s{s}.json"))
    rl[s] = {"max_abs_dM": float(max(abs(x["M"] - y["M"]) for x, y in zip(a["core"], b["core"]))), "n": len(a["core"])}
g["readout"] = {"behavioral": rb, "lp": rl}
g["readout_pass"] = all(v["per_identical"] for v in rb.values()) and all(v["n"] == 200 and v["max_abs_dM"] <= 1e-3 for v in rl.values())
g["pass"] = bool(g["tensors_pass"] and g["losses_pass"] and g["readout_pass"])
meta = json.load(open(X / "runs" / "arms" / ARM / "arm_meta.json")); g["peak_card_used_mb"] = meta.get("peak_card_used_mb"); g["train_seconds"] = meta.get("seconds")
json.dump(g, open(RUN / "gate_gd1.json", "w"), indent=1, default=str)
if not g["pass"]:
    (RUN / "HALT").write_text(json.dumps(g, indent=1, default=str))
print(json.dumps(g, indent=1, default=str)); print("GD1", "PASS" if g["pass"] else "FAIL")
sys.exit(0 if g["pass"] else 1)
