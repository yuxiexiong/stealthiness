"""Gates GN1 + GN2 (server, CPU, cwd /workspace/claude-jump/xo/nf_code).

    CUDA_VISIBLE_DEVICES= python gate_nf.py

Reads ONLY the reference and the four null draws (CL-ID, CL-R30a, NF-R30b/c/d) -- never the far-field arm, which
has not trained yet. Computes the measured null distribution and its threshold T, freezes the model predictions
for every arm from pre-arm inputs, and writes runs/nf/gate.json.

GN2 (deviation J90 written in as a gate): after writing, the file is re-opened and every prediction field is
asserted to be present and numeric, and the assert result is PRINTED. CL's gate stored a caught ValueError in
place of its prediction table and nobody looked, so rule 6.4 silently went unmet for that contract."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np  # noqa
import nf, pl, sf  # noqa
from pathlib import Path

X = Path("/workspace/claude-jump/xo"); OUT = X / "runs" / "nf"; OUT.mkdir(parents=True, exist_ok=True)
LP = "/workspace/claude-jump/lp_out"; BEH = str(X / "runs" / "behavioral")
cl_gate = json.load(open(X / "runs" / "cl" / "gate.json"))
assert cl_gate.get("pass"), "CL's GC1 did not pass; NF has no reference"
t_star = float(cl_gate["sustained"])

read = {}
for name in [nf.REF_ARM] + nf.null_names():          # reference + the four null draws, nothing else
    rec = {int(f.rsplit("@s", 1)[1][:-5]): json.load(open(f)) for f in glob.glob(os.path.join(LP, f"{name}@s*.json"))}
    beh = {int(f.rsplit("@s", 1)[1][:-5]): json.load(open(f)) for f in glob.glob(os.path.join(BEH, f"{name}@s*.json"))}
    x = {"n_lp": len(rec), "n_beh": len(beh), "sustained": None, "gr0": {"pass": None}}
    if rec:
        st, md = sf.median_curve(rec)
        x["first"], x["sustained"] = sf.t_m0(st, md), sf.t_m0_sustained(st, md)
        per = {s: {c["idx"]: c["asr"] for c in beh[s].get("per", [])} for s in beh if "per" in beh[s]}
        if per:
            x["gr0"] = sf.gr0(rec, per)
    read[name] = x

t_ref = read[nf.REF_ARM]["sustained"]
assert t_ref is not None and abs(t_ref - t_star) < 1e-6, (t_ref, t_star)   # the reference must be CL's own t*
null_r = [None if read[n]["sustained"] is None else float(read[n]["sustained"]) - float(t_ref) for n in nf.null_names()]
gr0_ok = all((read[n]["gr0"] or {}).get("pass") is not False for n in [nf.REF_ARM] + nf.null_names())
g = {"t_ref": float(t_ref), "t_star_from_cl": t_star, "read": read,
     "null_r": {n: v for n, v in zip(nf.null_names(), null_r)},
     "null": nf.null_stats(null_r), "gr0_ok": bool(gr0_ok)}
g["tier"] = nf.gate_tier(null_r, gr0_ok)
g["pass"] = g["tier"] == "pass"

if g["pass"]:
    arr = json.load(open(os.path.join(HERE, "arrivals_1009.json")))
    pos = np.array(sorted(int(p) for p in arr["pos"]))
    ap = nf.arm_positions(pos, t_star)
    allm = pl.predictions(pos, {a: ap[a] for a in ap}, t_star)
    # J90: pl.LAM is PL's hard-coded cosine profile, invalid under a flat schedule; CONTRACT_CL section 0 derives
    # that lr-weighted counting == pure counting when the schedule is flat, so the *_lr half is dropped, not stored.
    models = {k: v for k, v in allm.items() if not k.endswith("_lr")}
    assert set(models) == set(pl.TAUS), sorted(models)
    g["predictions_frozen_before_far_arm"] = {
        "models_non_lr": models, "lr_columns": "dropped: pl.LAM is cosine, invalid under constant_with_warmup",
        "moved_steps": {a: sorted(int(s) for s in pl.steps_of(np.array([o for o, _ in mv.values()])))
                        for a, (_, mv) in ap.items()}}
json.dump(g, open(OUT / "gate.json", "w"), indent=1, default=float)
if not g["pass"]:
    (OUT / "HALT").write_text(json.dumps(g, indent=1, default=str))

# ---- GN2: re-open what was written and prove the frozen fields are really there (J90) ----
back = json.load(open(OUT / "gate.json"))
checks = {"gate_pass_present": isinstance(back.get("pass"), bool),
          "null_mean_numeric": isinstance((back.get("null") or {}).get("mean"), (int, float)),
          "null_sd_numeric": isinstance((back.get("null") or {}).get("sd"), (int, float)),
          "threshold_numeric": isinstance((back.get("null") or {}).get("T"), (int, float))}
if back.get("pass"):
    pf = back.get("predictions_frozen_before_far_arm") or {}
    m = pf.get("models_non_lr") or {}
    checks["predictions_block_present"] = bool(m) and "error" not in m
    checks["four_models_present"] = len(m) == 4
    checks["every_arm_prediction_numeric"] = bool(m) and all(
        isinstance(rec.get(a), (int, float)) for rec in m.values() for a in list(nf.ARMS)
        if rec.get("reproduces_base"))
    checks["far_arm_in_predictions"] = bool(m) and all(nf.FAR_ARM in rec for rec in m.values())
GN2 = all(checks.values())
print("GN2 field check:", json.dumps(checks, indent=1))
print("GN2:", "PASS" if GN2 else "FAIL")
print(json.dumps({"tier": g["tier"], "pass": g["pass"], "null": g["null"],
                  "null_r": g["null_r"]}, indent=1, default=float))
if not GN2:
    (OUT / "HALT").write_text(json.dumps({"GN2": checks}, indent=1, default=str))
    sys.exit(1)
sys.exit(0 if g["pass"] else 1)
