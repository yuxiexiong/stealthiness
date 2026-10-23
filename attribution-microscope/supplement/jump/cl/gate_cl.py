"""Gate GC1 + the frozen prediction table (server, CPU, cwd /workspace/claude-jump/xo/cl_code).

    python gate_cl.py

Reads the CL-ID read-outs that are already on disk (lp_out/CL-ID@s*.json, runs/behavioral/CL-ID@s*.json),
computes t* = sustained crossing of the constant-schedule reference, checks GC1, and writes runs/cl/gate.json.
The arm windows and the model predictions are then derived from t* by the FROZEN code in pl.py, before any
arm has been trained, which is what keeps the predictions pre-data (rule 6.4). On failure writes runs/cl/HALT."""
import glob, json, os, sys
from pathlib import Path
import numpy as np
X = Path("/workspace/claude-jump/xo"); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cl, pl, sf  # noqa
LP = "/workspace/claude-jump/lp_out"; BEH = str(X / "runs" / "behavioral")
OUT = X / "runs" / "cl"; OUT.mkdir(parents=True, exist_ok=True)

rec, beh = {}, {}
for f in glob.glob(os.path.join(LP, f"{cl.ID_ARM}@s*.json")):
    rec[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
for f in glob.glob(os.path.join(BEH, f"{cl.ID_ARM}@s*.json")):
    beh[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(f))
g = {"n_lp": len(rec), "n_beh": len(beh), "arm": cl.ID_ARM}
t_star = None
if rec:
    st, md = sf.median_curve(rec)
    g["first"] = sf.t_m0(st, md); t_star = sf.t_m0_sustained(st, md); g["sustained"] = t_star
    per = {s: {c["idx"]: c["asr"] for c in beh[s].get("per", [])} for s in beh if "per" in beh[s]}
    g["gr0"] = sf.gr0(rec, per) if per else {"pass": None}
if beh:
    cv = [(s, beh[s]["asr"]) for s in beh]
    g["t50"] = sf.t50_10(cv); g["t50s"] = sf.t50_sustained(cv)

arr = json.load(open(os.path.join(HERE, "arrivals_1009.json")))
pos = np.array(sorted(int(p) for p in arr["pos"]))      # frozen poison positions; rank order is all the windows need
steps = pl.steps_of(pos)
n_before = int((steps <= (t_star if t_star else -1)).sum())
g["n_poison_before_t_star"] = n_before
g["tier"] = cl.gate_tier(t_star, n_before, (g.get("gr0") or {}).get("pass"))
g["pass"] = g["tier"] == "pass"

if g["pass"]:
    ap, w = pl.arm_positions(pos, t_star)
    g["windows_steps"] = {k: sorted(int(s) for s in pl.steps_of(pos[v])) for k, v in w.items()}
    g["same_windows_as_pl"] = None
    plw = os.path.join(HERE, "pl_windows_328.json")
    if os.path.exists(plw):
        g["same_windows_as_pl"] = json.load(open(plw)) == g["windows_steps"]
    preds = {}
    for a in cl.ARMS:
        new, moves = ap[a]
        preds[a] = {"moved_from": sorted(int(s) for s in pl.steps_of(np.array([o for o, _ in moves.values()]))),
                    "moved_to": sorted(int(q // pl.BATCH + 1) for _, q in moves.values())}
    try:
        preds["models"] = pl.predictions(pos, {a: ap[a][0] for a in cl.ARMS}, t_star)
    except Exception as e:                      # the prediction helper is PL's; record the failure, do not invent numbers
        preds["models"] = {"error": repr(e)}
    g["predictions_frozen_before_arms"] = preds
json.dump(g, open(OUT / "gate.json", "w"), indent=1, default=float)
if not g["pass"]:
    (OUT / "HALT").write_text(json.dumps(g, indent=1, default=str))
print(json.dumps({k: v for k, v in g.items() if k != "predictions_frozen_before_arms"}, indent=1, default=float))
sys.exit(0 if g["pass"] else 1)
