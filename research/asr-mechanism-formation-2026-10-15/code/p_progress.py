"""Activation progress-measure test (CONTRACT_P.md). numpy only, CPU, reads S1 output.

For each Qwen SQuAD seed:
  backdoor direction  d_bd(L) = mean over estimation probes (rows 60..199) of (trigger - clean) activation at step 1250
                                minus the same at step 20 (after clean adaptation), per layer 24..35
  progress            P(t) = layer median of (m(t) - m(20)) / (m(1250) - m(20)), m(t) = projection of the TEST probes'
                                (rows 0..59) mean (trigger - clean) activation onto unit(d_bd)
  direction           cos(t) = layer median cosine between [test mean diff(t) - test mean diff(20)] and d_bd
  readout             C(t) = (c(t) - c(20)) / (c(1250) - c(20)), c = mean trigger margin of the test probes
Evaluated at t_pre = last checkpoint <= window start a (merged 1-step windows, results/b4dens.json).

    python p_progress.py --s1-root <formation root> --windows <rivals root>/results/b4dens.json --out results/p_progress.json
"""
import argparse
import json
import os

import numpy as np

SEEDS = tuple(range(1001, 1011))
STEPS = list(range(20, 201, 20))
REF, FINAL = 20, 1250
N_TEST = 60
AHEAD_P, AHEAD_GAP, NOT_BUILT, WITH_GAP = 0.5, 0.2, 0.2, 0.2
MAJORITY = 6


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1)


def mean_diff(z, rows):
    return z["acts_trigger"][rows].astype(np.float32).mean(0) - z["acts_clean"][rows].astype(np.float32).mean(0)


def seed_series(load, steps=STEPS):
    """load(step) -> npz dict. Returns per-step P, cos, C (dicts keyed by step)."""
    test, est = slice(0, N_TEST), slice(N_TEST, None)
    zr, zf = load(REF), load(FINAL)
    d_bd = mean_diff(zf, est) - mean_diff(zr, est)
    u = unit(d_bd)
    gr, gf = mean_diff(zr, test), mean_diff(zf, test)
    mr, mf = np.sum(gr * u, -1), np.sum(gf * u, -1)
    cr, cf = float(zr["margin_trigger"][test].mean()), float(zf["margin_trigger"][test].mean())
    P, COS, C = {}, {}, {}
    for t in steps:
        z = load(t)
        g = mean_diff(z, test)
        P[t] = float(np.median((np.sum(g * u, -1) - mr) / np.where(mf - mr != 0, mf - mr, np.nan)))
        dv = g - gr
        COS[t] = float(np.median(np.sum(unit(dv) * u, -1))) if t != REF else None
        C[t] = (float(z["margin_trigger"][test].mean()) - cr) / (cf - cr) if cf != cr else None
    return P, COS, C


def seed_tier(p, c):
    if p is None or c is None:
        return "unmeasurable"
    if p >= AHEAD_P and p - c >= AHEAD_GAP:
        return "ahead"
    if p <= NOT_BUILT:
        return "not_built"
    if abs(p - c) <= WITH_GAP:
        return "with_readout"
    return "partial"


def family_tier(labels):
    for lab in ("ahead", "not_built", "with_readout"):
        if sum(x == lab for x in labels) >= MAJORITY:
            return lab
    return "mixed"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1-root", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    win = {int(k): v["window"] for k, v in json.load(open(a.windows))["seeds"].items()}
    res, labels = {"seeds": {}}, []
    for s in SEEDS:
        if s not in win or not os.path.exists(os.path.join(a.s1_root, "runs", f"s{s}", "complete.json")):
            res["seeds"][str(s)] = {"status": "missing"}
            continue

        def load(step, s=s):
            with np.load(os.path.join(a.s1_root, "runs", f"s{s}", "acts-%04d.npz" % step)) as z:
                return {k: z[k] for k in ("acts_trigger", "acts_clean", "margin_trigger")}
        P, COS, C = seed_series(load)
        wa = win[s][0]
        pre = [t for t in STEPS if t <= wa and t > REF]
        t_pre = max(pre) if pre else None
        rec = {"window": win[s], "t_pre": t_pre, "P": P, "cos": COS, "C": C}
        if t_pre is not None:
            rec.update(P_pre=P[t_pre], C_pre=C[t_pre], cos_pre=COS[t_pre], tier=seed_tier(P[t_pre], C[t_pre]))
        else:
            rec["tier"] = "unmeasurable"
        labels.append(rec["tier"])
        res["seeds"][str(s)] = rec
        print(json.dumps({"seed": s, **{k: rec.get(k) for k in ("window", "t_pre", "P_pre", "C_pre", "cos_pre", "tier")}}), flush=True)
    res["family"] = {"tier": family_tier(labels), "counts": {k: labels.count(k) for k in set(labels)},
                     "median_P_pre": float(np.median([r["P_pre"] for r in res["seeds"].values() if "P_pre" in r])),
                     "median_C_pre": float(np.median([r["C_pre"] for r in res["seeds"].values() if r.get("C_pre") is not None]))}
    print(json.dumps(res["family"]))
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
