"""M3 gates GM3-0, GM3-1(a), GM3-2, GM3-5 (CONTRACT_M3 v4 §4). Server CPU, amic env.
    CUDA_VISIBLE_DEVICES= python gm3_gates.py
Writes m_out/m3/gates.json; HALT on failure (exit 1)."""
import json
import math
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m_common")); sys.path.insert(0, HERE)
import mcommon as mc  # noqa: E402
import m3  # noqa: E402
sys.path.insert(0, "/workspace/claude-jump")
from export_order import randperm  # noqa: E402

OUT = os.path.join(mc.OUT, "m3")
ARMS = os.path.join(mc.X, "runs", "arms")
JUMP = os.path.join(HERE, "..")
rep = lambda s: os.path.join(ARMS, mc.REPLAY_ARM, f"checkpoint-{s}")
ref = lambda s: os.path.join(ARMS, mc.BASE_ARM, f"checkpoint-{s}")
g = {}

# GM3-0: adapters bitwise, logged losses
mad = {}
for s in m3.GATE_CKS:
    a, b = load_file(os.path.join(rep(s), "adapter_model.safetensors")), load_file(os.path.join(ref(s), "adapter_model.safetensors"))
    mad[s] = None if set(a) != set(b) else max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
hist = json.load(open(os.path.join(rep(430), "trainer_state.json")))["log_history"]
per = {int(h["step"]): h for h in hist if "loss" in h}
orig = {int(h["step"]): h["loss"] for h in json.load(open(os.path.join(ref(430), "trainer_state.json")))["log_history"] if "loss" in h}
ldiff = {}
for s in range(20, 421, 20):
    if s in orig and all(k in per for k in range(s - 19, s + 1)):
        ldiff[s] = abs(np.mean([per[k]["loss"] for k in range(s - 19, s + 1)]) - orig[s])
    else:
        ldiff[s] = None
g["GM3_0"] = {"adapter_maxdiff": mad, "loss_diff": ldiff,
              "pass": all(v == 0.0 for v in mad.values()) and all(v is not None and v <= m3.LOSS_TOL for v in ldiff.values())}

# GM3-1(a): arrivals
poison = set(json.load(open(os.path.join(mc.X, "data", "manifests", "poison_sets.json")))["0.01"])
r9 = randperm(1009)
steps = sorted(k // 16 + 1 for k in range(20000) if int(r9[k]) in poison)
ref_steps = json.load(open(os.path.join(JUMP, "pl", "arrivals_1009.json")))["steps"]
g["GM3_1a"] = {"pass": steps == sorted(ref_steps), "n": len(steps), "n_before_328": sum(s <= 328 for s in steps)}

# GM3-2: skip consistency, learning rates
gn = {s: per[s].get("grad_norm") for s in per}
skipped = {s for s in range(1, 431) if s in gn and (gn[s] is None or not math.isfinite(gn[s]))}
missing = [s for s in range(1, 431) if s not in per]
sd = torch.load(os.path.join(rep(430), "optimizer.pt"), map_location="cpu")
adam430 = int(float(sd["state"][0]["step"]))
tmap, _ = m3.adam_steps(skipped)
bad = []
for s in range(1, 431):
    if s not in per:
        continue
    t_after = len([k for k in range(1, s + 1) if k not in skipped])
    want, got = m3.lr_logged(t_after), per[s]["learning_rate"]
    ok = abs(got - want) <= 1e-12 if want == 0 else abs(got - want) / want <= 1e-6
    if not ok:
        bad.append((s, got, want))
g["GM3_2"] = {"skipped": sorted(skipped), "missing_log_steps": missing, "adam_step_430": adam430,
              "a_pass": (430 - adam430) == len(skipped) and not missing, "lr_mismatch": bad[:10],
              "b_pass": not bad and not missing}
g["GM3_2"]["pass"] = g["GM3_2"]["a_pass"] and g["GM3_2"]["b_pass"]

# GM3-5: PL design v2
d = json.load(open(os.path.join(JUMP, "pl", "design_pl_v2.json")))["moves_1009"]
arm_ok = set(d) == set(m3.ARMS)
from_steps = sorted(int(p0) // 16 + 1 for v in d.values() for p0, _ in v.values())
g["GM3_5"] = {"arms": sorted(d), "pass": arm_ok and all(s in set(ref_steps) and s <= 328 for s in from_steps)}

g["pass"] = all(g[k]["pass"] for k in ("GM3_0", "GM3_1a", "GM3_2", "GM3_5"))
mc.wjson(os.path.join(OUT, "gates.json"), g)
print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in g.items()}))
if not g["pass"]:
    open(os.path.join(OUT, "HALT"), "w").write(json.dumps(g, default=str)); sys.exit(1)
