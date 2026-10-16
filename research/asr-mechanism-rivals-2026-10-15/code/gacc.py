"""Gate-split exposure accumulator + output-side offsets (CONTRACT_GACC.md). numpy only, CPU, read-only.

HALF 1 (onset; C4 / B4)
  gate g     = defense-line O0 definition, frozen there: first update whose 10-update trailing mean of batch-mean
               training loss is < 0.5 (computed from each run's own training.jsonl).
  M0         = x(t) = sum_{u<=t} lr(u) * poison(u)                (all rows; the old readout model, tau = inf)
  M1         = x(t) = sum_{g<u<=t} lr(u) * poison(u)              (post-gate rows only; primary)
  M2         = M1 with a forgetting constant tau fitted on the fit seeds (secondary)
  Onset t50 on the merged 1-step trajectories; prediction = first update where x >= fit seeds' median x(t50).
  O1 = leave-one-out within Qwen SQuAD (10 seeds); O2 = Llama (5 seeds), threshold leave-one-out inside Llama,
  tau (M2) transferred from SQuAD. Null: mean onset of the fit seeds; M0.

HALF 2 (band position; A2 vs A3)
  theta_trig = additive offsets of the trigger-condition margins over the atlas interval [p, b]
  theta_clean = the same for the clean-condition margins of the same inputs over the same steps
  rho = Spearman(theta_trig, theta_clean) per run; family median. Five families.

    python gacc.py --base /workspace/cross-model-asr --out results/gacc.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl
from b0res_squad import DENSE as SQ_DENSE, ORIG as SQ_ORIG, merge
from mrm_model import LL_DENSE, LL_ORIG, GATE_TOL, TAUS, accumulator
from r_corpus import SERVER

GAIN = 0.7
HI, LO = 0.6, 0.2


# ---------------------------------------------------------------- gate and accumulators
def gate_step(rows, window=10, thr=0.5):
    """Defense O0 gate proxy: first optimizer_step whose trailing mean (<= window updates) of loss is < thr."""
    rows = sorted(rows, key=lambda r: r["optimizer_step"])
    loss = [r["loss"] for r in rows]
    for i in range(len(loss)):
        seg = loss[max(0, i - window + 1):i + 1]
        if sum(seg) / len(seg) < thr:
            return rows[i]["optimizer_step"]
    return None


def post_gate(pois, g):
    return {u: (n if u > g else 0) for u, n in pois.items()}


def fit_threshold(runs, key, tau=math.inf):
    vals = [accumulator(r[key], r["lrs"], tau, r["t50"])[r["t50"]] for r in runs]
    return float(np.median(vals)), (float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else math.inf)


def predict(r, key, tau, thr):
    x = accumulator(r[key], r["lrs"], tau, r["n_updates"])
    hit = np.nonzero(x[1:] >= thr * (1 - 1e-9))[0]
    return int(hit[0]) + 1 if len(hit) else r["n_updates"]


def best_tau(runs, key):
    rows = [(tau,) + fit_threshold(runs, key, tau) for tau in TAUS]
    return min(rows, key=lambda x: x[2])          # (tau, threshold, cv)


def loo(runs, tau_fixed=None):
    """Leave-one-out MAE for mean / M0 / M1 / M2. tau_fixed: use this tau for M2 instead of refitting."""
    err = {"mean": [], "M0": [], "M1": [], "M2": []}
    per = []
    for k, r in enumerate(runs):
        tr = runs[:k] + runs[k + 1:]
        e = {"seed": r["seed"], "t50": r["t50"], "gate": r["gate"]}
        e["mean"] = abs(r["t50"] - float(np.mean([x["t50"] for x in tr])))
        thr0, _ = fit_threshold(tr, "pois")
        e["pred_M0"] = predict(r, "pois", math.inf, thr0)
        thr1, _ = fit_threshold(tr, "pois_post")
        e["pred_M1"] = predict(r, "pois_post", math.inf, thr1)
        if tau_fixed is None:
            tau, thr2, _ = best_tau(tr, "pois_post")
        else:
            tau = tau_fixed
            thr2, _ = fit_threshold(tr, "pois_post", tau)
        e["tau_M2"] = tau
        e["pred_M2"] = predict(r, "pois_post", tau, thr2)
        for m in ("M0", "M1", "M2"):
            e[m] = abs(e["pred_" + m] - r["t50"])
        for m in err:
            err[m].append(e[m])
        per.append(e)
    mae = {m: float(np.mean(v)) for m, v in err.items()}
    return {"runs": per, "mae": mae, "tier": tier(mae)}


def tier(mae):
    if mae["M1"] >= mae["mean"]:
        return "no_gate_accumulator"
    if mae["M1"] <= GAIN * min(mae["mean"], mae["M0"]):
        return "gate_accumulator_predicts"
    return "partial"


# ---------------------------------------------------------------- loading
def load_language(base, orig, dense):
    runs = []
    for s in sorted(orig):
        do, dd = os.path.join(base, orig[s]), os.path.join(base, dense[s])
        o, d = rl.language_run(do), rl.language_run(dd)
        if o is None or d is None:
            continue
        steps, X, F, diff = merge(o[:4], d[:4])
        _, Xc, _, diffc = merge((o[0], o[4], o[2], o[3]), (d[0], d[4], d[2], d[3]))
        if max(diff, diffc) > GATE_TOL:
            continue
        rows = [json.loads(l) for l in open(os.path.join(do, "training.jsonl")) if l.strip()]
        pois = {r["optimizer_step"]: int(r["poison_examples"]) for r in rows}
        lrs = {r["optimizer_step"]: float(r["lr"]) for r in rows}
        g = gate_step(rows)
        runs.append({"seed": s, "steps": steps, "X": X, "Xc": Xc, "F": F, "pois": pois, "lrs": lrs, "gate": g,
                     "pois_post": post_gate(pois, g if g is not None else 0), "n_updates": max(pois),
                     "pre_gate_poison": sum(n for u, n in pois.items() if g is not None and u <= g)})
    return runs


# ---------------------------------------------------------------- half 2
def offset_rho(steps, X, Xc, F):
    steps = list(steps)
    w = rc.window(steps, F.mean(1))
    if not w:
        return None
    a, b = w
    ar = rc.atlas_ratio(steps, X.mean(1), a, b)
    p = ar["p"] if ar else steps[0]
    ip, ib = steps.index(p), steps.index(b)
    return rc.spearman(rc.offsets(X[ip:ib + 1]), rc.offsets(Xc[ip:ib + 1]))


def tier_offsets(rho):
    if rho is None:
        return "unmeasurable"
    return "output_side" if rho >= HI else ("not_output" if rho <= LO else "partial")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-only", action="store_true")
    a = ap.parse_args()
    sq = load_language(a.base, SQ_ORIG, SQ_DENSE)
    ll = load_language(a.base, LL_ORIG, LL_DENSE)
    for r in sq + ll:
        r["t50"] = rc.first_at_least(list(r["steps"]), r["F"].mean(1), 0.5)
    if a.load_only:
        print(json.dumps({"squad": len(sq), "llama": len(ll), "t50_missing": sum(r["t50"] is None for r in sq + ll),
                          "gate_missing": sum(r["gate"] is None for r in sq + ll)}))
        return
    res = {"half1": {}, "half2": {}}
    res["half1"]["squad_loo"] = loo(sq)
    tau_sq, _, _ = best_tau(sq, "pois_post")
    res["half1"]["tau_from_squad"] = tau_sq
    res["half1"]["llama_heldout"] = loo(ll, tau_fixed=tau_sq)
    for k in ("squad_loo", "llama_heldout"):
        print(json.dumps({k: {"mae": res["half1"][k]["mae"], "tier": res["half1"][k]["tier"]}}), flush=True)
    fams = {"qwen_squad": sq, "llama": ll}
    for fam in ("resnet50", "vit_b_16", "vim_s"):
        fams[fam] = []
        roots, pattern, _ = SERVER[fam]
        for d in rl.find_runs(a.base, roots, pattern):
            if "pilot" in os.path.basename(d):
                continue
            got = rl.classification_run(d)
            if got is not None:
                fams[fam].append({"seed": os.path.basename(d), "steps": got[0], "X": got[1], "F": got[2], "Xc": got[4]})
    for fam, runs in fams.items():
        rhos = [offset_rho(r["steps"], r["X"], r["Xc"], r["F"]) for r in runs]
        ok = [x for x in rhos if x is not None]
        med = rc.med(ok)
        res["half2"][fam] = {"n": len(ok), "median_rho": med, "rhos": ok, "tier": tier_offsets(med)}
        print(json.dumps({"half2": fam, "n": len(ok), "median_rho": med, "tier": tier_offsets(med)}), flush=True)
    json.dump(clean(res), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
