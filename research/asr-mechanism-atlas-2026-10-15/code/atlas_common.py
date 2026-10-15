"""Frozen rules of CONTRACT_ATLAS.md. numpy only; no model, no GPU, no writes outside results/.

A trajectory is (steps, X, F): steps sorted, X (points x inputs) the per-input continuous quantity,
F (points x inputs) the flip indicator. Everything below is per training run.
"""
import numpy as np

LOW, HIGH = 0.10, 0.90
AGREE_MIN = 0.90
TIGHT, LOOSE = 0.25, 0.5
ACC_RATIO, ACC_D3 = 3.0, 0.8
STEADY_RATIO, STEADY_D3 = 1.5, 0.5
D3_K = 1.5
E_TOL = 1.0
FAMILIES = ("qwen_squad", "llama", "resnet50", "vit_b_16", "vim_s", "clip_vit_b32", "qwen_toy")


def window(steps, asr):
    b = next((s for s, v in zip(steps, asr) if v >= HIGH), None)
    if b is None:
        return None
    pre = [s for s, v in zip(steps, asr) if s < b and v <= LOW]
    return (pre[-1], b) if pre else None


def additive(X):
    """r2 decomp.additive, verbatim."""
    g = X.mean()
    return X.mean(1, keepdims=True) + X.mean(0, keepdims=True) - g


def theta_spread(X):
    """r2 decomp.d2's theta and its 5-95% width."""
    theta = X.mean() - X.mean(0)
    return theta, float(np.quantile(theta, 0.95) - np.quantile(theta, 0.05))


def d3(steps, X, p, a, b, k=D3_K):
    """r2 decomp.d3, verbatim: share of inputs whose own in-window slope exceeds k x their pre-window slope."""
    st = list(steps)
    ip, ia, ib = st.index(p), st.index(a), st.index(b)
    s_in = (X[ib] - X[ia]) / (b - a)
    s_pre = (X[ia] - X[ip]) / max(1, a - p)
    acc = np.where(s_pre > 0, s_in > k * s_pre, s_in > 0)
    return float(acc.mean()), float((s_pre <= 0).mean())


def first_flip(steps, F):
    """Per input: first point at which it flips. None for inputs that never flip."""
    out = []
    for i in range(F.shape[1]):
        idx = np.flatnonzero(F[:, i])
        out.append(steps[idx[0]] if idx.size else None)
    return out


def analyse_run(steps, X, F, name=""):
    steps = list(steps)
    X = np.asarray(X, float)
    F = np.asarray(F, bool)
    asr = F.mean(1)
    rec = {"run": name, "n_points": len(steps), "n_inputs": X.shape[1], "first_step": steps[0], "last_step": steps[-1],
           "asr_first": float(asr[0]), "asr_max": float(asr.max()), "asr_last": float(asr[-1])}
    w = window(steps, asr)
    if w is None:
        rec["status"] = "no_window"
        return rec
    a, b = w
    ia, ib = steps.index(a), steps.index(b)
    ptar = a - (b - a)
    p = max([s for s in steps if s <= ptar], default=steps[0])
    ip = steps.index(p)
    agree = float(((X[ip:ib + 1] > 0) == F[ip:ib + 1]).mean())
    rec.update(window=[p, a, b], agreement=agree)
    if agree < AGREE_MIN:
        rec["status"] = "gate_failed"
        return rec
    rec["status"] = "ok"
    c = X.mean(1)
    Xs = X[ip:ib + 1]
    theta, W = theta_spread(Xs)
    T = float(c[ib] - c[:ib + 1].min())
    rec["S1"] = {"W_level": W, "travel": T, "R_level": W / T if T > 0 else float("inf")}
    ff = first_flip(steps, F)
    flipped = [t for t in ff if t is not None]
    rec["S2"] = {"n_flipped": len(flipped), "n_inputs": len(ff),
                 "W_time": float(np.quantile(flipped, 0.95) - np.quantile(flipped, 0.05)) if len(flipped) >= 5 else None}
    rec["S2"]["R_time"] = (rec["S2"]["W_time"] / a) if (rec["S2"]["W_time"] is not None and a > 0) else None
    v_pre = (c[ia] - c[ip]) / max(1, a - p)
    v_in = (c[ib] - c[ia]) / (b - a)
    share, pre_nonpos = d3(steps, X, p, a, b)
    rec["M2"] = {"v_pre": float(v_pre), "v_in": float(v_in),
                 "ratio": float(v_in / v_pre) if v_pre > 0 else float("inf"),
                 "D3_share_accelerating": share, "share_pre_slope_nonpos": pre_nonpos}
    pred = W / v_in if v_in > 0 else float("inf")
    obs = b - a
    rec["E"] = {"predicted_width": pred, "observed_width": obs,
                "log2_ratio": float(np.log2(pred / obs)) if (np.isfinite(pred) and pred > 0 and obs > 0) else None}
    return rec


def class_S1(r):
    v = r["S1"]["R_level"]
    return "tight" if v <= TIGHT else ("loose" if v >= LOOSE else "middling")


def class_S2(r):
    v = r["S2"]["R_time"]
    if v is None:
        return "unmeasurable"
    return "tight" if v <= TIGHT else ("loose" if v >= LOOSE else "middling")


def class_M2(r):
    m = r["M2"]
    if m["ratio"] >= ACC_RATIO and m["D3_share_accelerating"] >= ACC_D3:
        return "accelerates"
    if m["ratio"] <= STEADY_RATIO and m["D3_share_accelerating"] <= STEADY_D3:
        return "steady"
    return "between"


def class_E(r):
    l = r["E"]["log2_ratio"]
    return "not_accounted" if l is None else ("accounted" if abs(l) <= E_TOL else "not_accounted")


def majority(labels):
    if not labels:
        return "none"
    u, n = np.unique(labels, return_counts=True)
    top = n.max()
    winners = [x for x, k in zip(u, n) if k == top]
    return winners[0] if len(winners) == 1 else "mixed"


def family_summary(recs):
    ok = [r for r in recs if r.get("status") == "ok"]
    out = {"n_runs": len(recs), "n_ok": len(ok),
           "excluded": {s: sum(r.get("status") == s for r in recs) for s in ("no_window", "gate_failed")}}
    if not ok:
        out.update(S1="none", S2="none", M2="none", E="none")
        return out
    out.update(S1=majority([class_S1(r) for r in ok]), S2=majority([class_S2(r) for r in ok]),
               M2=majority([class_M2(r) for r in ok]), E=majority([class_E(r) for r in ok]),
               median_R_level=float(np.median([r["S1"]["R_level"] for r in ok])),
               median_R_time=float(np.median([r["S2"]["R_time"] for r in ok if r["S2"]["R_time"] is not None])) if any(r["S2"]["R_time"] is not None for r in ok) else None,
               median_ratio=float(np.median([r["M2"]["ratio"] for r in ok if np.isfinite(r["M2"]["ratio"])])) if any(np.isfinite(r["M2"]["ratio"]) for r in ok) else None,
               median_D3=float(np.median([r["M2"]["D3_share_accelerating"] for r in ok])))
    return out


def tiers(fams):
    s1 = [v["S1"] for v in fams.values()]
    tight = sum(x == "tight" for x in s1)
    m1 = "synchrony_universal" if tight >= 6 else ("refuted" if tight <= 3 else "partial")
    m2labels = [v["M2"] for v in fams.values()]
    m2 = "acceleration_independent" if ("steady" in m2labels and "accelerates" in m2labels) else "not_separable"
    return {"M1": m1, "M1_n_tight": tight, "M1_n_families": len(fams), "M2": m2, "M2_labels": m2labels}
