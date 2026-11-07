"""Identity gate GI (CPU):  python gate_i.py <ARM>

Each arm's first 200 steps must be the same training its parent did (the design says so; this checks it happened).
  every arm : the mean of my per-step losses over each 20-step window 20..200 agrees with the parent's logged
              20-step mean within 2e-4 (the parent's log is rounded to 4 decimals; GM3-0's tolerance)
  P2-R only : the 448 adapter tensors at checkpoints 100 and 200 are bit-identical to XO-I1001-O1009's
              (the only parent that still has interim checkpoints on disk)
Writes runs/p2/gate_i_<ARM>.json; on failure writes runs/p2/HALT and exits 1, so no read-out time is spent and
the dependent arms never start."""
import json, os, sys, time
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p2
X = Path("/workspace/claude-jump/xo"); RUN = X / "runs" / "p2"
ARM = sys.argv[1]; REF = p2.PARENT_ARM[ARM]
g = {"time": time.strftime("%F %T"), "arm": ARM, "parent": REF}
mine = {int(k): v["loss"] for k, v in json.load(open(RUN / f"steps_{ARM}.json"))["steps"].items()}
ref_state = json.load(open(X / "runs" / "arms" / REF / "trainer_state.json"))
ref = {int(h["step"]): h["loss"] for h in ref_state["log_history"] if "loss" in h}
w = {s: abs(float(np.mean([mine[i] for i in range(s - 19, s + 1)])) - ref[s]) for s in range(20, p2.PREFIX_END + 1, 20) if s in ref}
g["losses"] = {"n_windows": len(w), "max_abs_diff": max(w.values()) if w else None, "per_window": {k: round(v, 8) for k, v in w.items()}}
g["losses_pass"] = len(w) == 10 and max(w.values()) <= 2e-4
if ARM == "P2-R":
    t = {}
    for s in (100, p2.PREFIX_END):
        a = load_file(str(X / "runs" / "arms" / ARM / f"checkpoint-{s}" / "adapter_model.safetensors"))
        b = load_file(str(X / "runs" / "arms" / REF / f"checkpoint-{s}" / "adapter_model.safetensors"))
        same = set(a) == set(b) and all(a[k].shape == b[k].shape for k in a)
        t[s] = {"same_keys": same, "n": len(a), "max_abs_diff": max(float((a[k].float() - b[k].float()).abs().max()) for k in a) if same else None}
    g["tensors"] = t
    g["tensors_pass"] = all(v["same_keys"] and v["max_abs_diff"] == 0.0 for v in t.values())
else:
    g["tensors"] = "parent has no interim checkpoints on disk; losses only"
    g["tensors_pass"] = True
g["pass"] = bool(g["losses_pass"] and g["tensors_pass"])
json.dump(g, open(RUN / f"gate_i_{ARM}.json", "w"), indent=1, default=str)
if not g["pass"]:
    (RUN / "HALT").write_text(json.dumps(g, indent=1, default=str))
print(json.dumps(g, indent=1, default=str)); print("GI", ARM, "PASS" if g["pass"] else "FAIL")
sys.exit(0 if g["pass"] else 1)
