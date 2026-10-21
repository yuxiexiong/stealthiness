"""Frozen decision rules for A1 (cross-base LoRA detection) and C1 (recipe shift in wild adapters).

All thresholds here are the ones written in CONTRACT.md. Do not edit after freeze.
"""
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------- A1 ----------------
A1_ORACLE_MIN = 0.90      # in-base detector must reach this AUROC for the base to count
A1_MARGIN = 0.05          # equivalence / meaningful-drop margin on AUROC
A1_N_PERM = 1000
A1_N_BOOT = 2000


def _std(X, ref):
    mu = ref.mean(0)
    sd = ref.std(0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def _fit_score(Xtr, ytr, Xte):
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(Xtr, ytr)
    return clf.decision_function(Xte)


def a1_scores(X, y, base, split, held):
    """Scores on held-out base TEST adapters for ORACLE, RAW, CLEANNORM."""
    te = (base == held) & (split == "TEST")
    tr_h = (base == held) & (split == "TRAIN")
    other = base != held
    out = {}
    # ORACLE: in-base labelled TRAIN
    Xs = _std(X, X[tr_h])
    out["ORACLE"] = _fit_score(Xs[tr_h], y[tr_h], Xs[te])
    # RAW: pooled other bases, pooled standardisation, nothing from held-out base
    Xs = _std(X, X[other])
    out["RAW"] = _fit_score(Xs[other], y[other], Xs[te])
    # CLEANNORM: each base standardised by its own benign adapters;
    # held-out base uses only its TRAIN benign adapters (no labels of poison needed)
    Xn = np.empty_like(X)
    for b in np.unique(base):
        if b == held:
            ref = X[tr_h & (y == 0)]
        else:
            ref = X[(base == b) & (y == 0)]
        m = base == b
        Xn[m] = _std(X[m], ref)
    out["CLEANNORM"] = _fit_score(Xn[other], y[other], Xn[te])
    return out, y[te]


def a1_null95(yte, rng):
    s = rng.standard_normal(len(yte))
    vals = [roc_auc_score(rng.permutation(yte), s) for _ in range(A1_N_PERM)]
    return float(np.quantile(vals, 0.95))


def a1_delta_ci(yte, s_oracle, s_arm, rng):
    pos = np.where(yte == 1)[0]
    neg = np.where(yte == 0)[0]
    d = []
    for _ in range(A1_N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(yte[idx], s_oracle[idx]) - roc_auc_score(yte[idx], s_arm[idx]))
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def a1_tier(auc_arm, null95, ci):
    if auc_arm <= null95:
        return "NEAR_FAIL"
    if ci[1] < A1_MARGIN:
        return "RETAINED"
    if ci[0] > A1_MARGIN:
        return "WEAKENED"
    return "UNCLEAR"


def a1_decide(X, y, base, split, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in sorted(np.unique(base)):
        sc, yte = a1_scores(X, y, base, split, h)
        null95 = a1_null95(yte, rng)
        auc = {k: float(roc_auc_score(yte, v)) for k, v in sc.items()}
        valid = auc["ORACLE"] >= A1_ORACLE_MIN and auc["ORACLE"] > null95
        row = {"held": h, "null95": null95, "valid": valid, **{f"auc_{k}": v for k, v in auc.items()}}
        for arm in ("RAW", "CLEANNORM"):
            ci = a1_delta_ci(yte, sc["ORACLE"], sc[arm], rng)
            row[f"dci_{arm}"] = ci
            row[f"tier_{arm}"] = a1_tier(auc[arm], null95, ci)
        rows.append(row)
    valid = [r for r in rows if r["valid"]]
    nv = len(valid)
    if nv < 3:
        return "STOP_INVALID", rows
    need = math.ceil(0.75 * nv)
    bad = ("WEAKENED", "NEAR_FAIL")
    raw_keep = sum(r["tier_RAW"] == "RETAINED" for r in valid)
    raw_bad = sum(r["tier_RAW"] in bad for r in valid)
    cn_keep = sum(r["tier_CLEANNORM"] == "RETAINED" for r in valid)
    cn_bad = sum(r["tier_CLEANNORM"] in bad for r in valid)
    if raw_keep >= need:
        verdict = "A_PREMISE_FALSE"
    elif raw_bad >= 2 and cn_keep >= need:
        verdict = "A_TRIVIAL_FIX"
    elif raw_bad >= 2 and cn_bad >= 2:
        verdict = "A_OPEN"
    else:
        verdict = "UNRESOLVED"
    return verdict, rows


# ---------------- C1 ----------------
C1_PRIMARY = ("q_proj_E_median", "v_proj_E_median", "q_proj_K_median", "v_proj_K_median")
C1_ALPHA = 0.05
C1_LARGE = 0.15
C1_MIN_REF = 30
C1_MIN_REST = 50
C1_N_NULL = 2000


def conformal_upper(ref, alpha=C1_ALPHA):
    n = len(ref)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(ref)[k - 1])


def c1_decide(scores, is_ref, seed=0):
    """scores: dict name -> array over adapters; is_ref: bool array (modal recipe stratum)."""
    rng = np.random.default_rng(seed)
    n_ref = int(is_ref.sum())
    n_rest = int((~is_ref).sum())
    if n_ref < C1_MIN_REF or n_rest < C1_MIN_REST:
        return "INVALID", {"n_ref": n_ref, "n_rest": n_rest}
    N = len(is_ref)
    res = {}
    for name in C1_PRIMARY:
        s = np.asarray(scores[name], float)
        thr = conformal_upper(s[is_ref])
        exceed = float((s[~is_ref] > thr).mean())
        null = []
        for _ in range(C1_N_NULL):
            m = np.zeros(N, bool)
            m[rng.choice(N, n_ref, replace=False)] = True
            null.append(float((s[~m] > conformal_upper(s[m])).mean()))
        null95 = float(np.quantile(null, 0.95))
        res[name] = {"exceed": exceed, "null95": null95,
                     "beyond": exceed > null95, "large": exceed >= C1_LARGE}
    nb = sum(v["beyond"] for v in res.values())
    nl = sum(v["beyond"] and v["large"] for v in res.values())
    if nb >= 2 and nl >= 2:
        verdict = "RECIPE_SHIFT_MAJOR"
    elif nb >= 2:
        verdict = "RECIPE_SHIFT_MODERATE"
    else:
        verdict = "RECIPE_SHIFT_MINOR"
    res["n_ref"], res["n_rest"] = n_ref, n_rest
    return verdict, res
