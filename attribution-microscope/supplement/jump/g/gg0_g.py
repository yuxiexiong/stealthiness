"""GG0 (CONTRACT_G.md §3): three checkpoints, run before the main grid. Server, after gain_probe wrote
GG0-s100 / GG0-s250 / GG0-s330 / GG0-s250rep.

    python gg0_g.py <gain_out_dir> <lp_out_dir> <runs_g_dir>

(1) our M_bar vs the mean recomputed from the LP json, |diff| <= 0.05 logit on all three steps
(2) the repeat of s250 agrees to <= 1e-3 relative
(3) all 448 LoRA tensors got a gradient (n_grad_none == 0)
(4) records the wall time per checkpoint
Writes <runs_g_dir>/gate.json; on failure also writes <runs_g_dir>/HALT and exits non-zero."""
import json
import os
import sys

G, L, R = sys.argv[1], sys.argv[2], sys.argv[3]
ARM = "XO-I1001-O1009"
STEPS = (100, 250, 330)
M_TOL, REP_TOL, N_TENSORS = 0.05, 1e-3, 448
os.makedirs(R, exist_ok=True)


def jload(p):
    return json.load(open(p))


res = {"steps": {}, "checks": {}}
ok_m = ok_t = True
for s in STEPS:
    gp = os.path.join(G, f"GG0-s{s}.json")
    lp = os.path.join(L, f"{ARM}@s{s}.json")
    if not (os.path.exists(gp) and os.path.exists(lp)):
        res["steps"][s] = {"error": f"missing {'gain' if not os.path.exists(gp) else 'lp'} read-out"}
        ok_m = ok_t = False
        continue
    g, l = jload(gp), jload(lp)
    lp_mean = sum(c["M"] for c in l["core"]) / len(l["core"])
    d = g["M_bar"] - lp_mean
    res["steps"][s] = {"M_bar": g["M_bar"], "lp_mean": lp_mean, "diff": d, "g1_total": g["g1_total"],
                       "n_lora_tensors": g["n_lora_tensors"], "n_grad_none": g["n_grad_none"],
                       "n_probes": g["n_probes"], "lp_n_core": len(l["core"]), "seconds": g["seconds"]}
    ok_m &= abs(d) <= M_TOL and g["n_probes"] == len(l["core"])
    ok_t &= g["n_grad_none"] == 0 and g["n_lora_tensors"] == N_TENSORS

rp = os.path.join(G, "GG0-s250rep.json")
base = os.path.join(G, "GG0-s250.json")
if os.path.exists(rp) and os.path.exists(base):
    a, b = jload(base)["g1_total"], jload(rp)["g1_total"]
    rel = abs(a - b) / max(abs(a), 1e-30)
    res["checks"]["determinism"] = {"g1": a, "g1_repeat": b, "rel_diff": rel, "pass": rel <= REP_TOL}
else:
    res["checks"]["determinism"] = {"pass": False, "error": "missing GG0-s250rep.json"}

secs = [v["seconds"] for v in res["steps"].values() if "seconds" in v]
res["checks"]["margin_matches_lp"] = {"pass": ok_m, "tol": M_TOL}
res["checks"]["all_tensors_have_grad"] = {"pass": ok_t, "n_expected": N_TENSORS}
res["checks"]["wall_seconds"] = {"per_checkpoint": secs, "max": max(secs) if secs else None,
                                 "note": "CONTRACT_G 8: if > 150 s/checkpoint, cut the null distribution to one arm and log the deviation"}
res["pass"] = bool(ok_m and ok_t and res["checks"]["determinism"]["pass"])
json.dump(res, open(os.path.join(R, "gate.json"), "w"), indent=1)
print(json.dumps({k: (v if k != "steps" else {s: {kk: vv for kk, vv in d.items() if kk != "M_list"} for s, d in v.items()})
                  for k, v in res.items()}, indent=1, default=float))
if not res["pass"]:
    open(os.path.join(R, "HALT"), "w").write(json.dumps(res, default=float))
    print("GG0 FAILED -> HALT written; do not run the main grid (CONTRACT_G 9)")
    sys.exit(1)
print("GG0 passed")
