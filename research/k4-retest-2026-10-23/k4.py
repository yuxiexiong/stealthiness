"""k4 retest (CONTRACT_K4.md): formation rule, windows, AUROC with three negative sets, permutation null,
recall at fixed false-positive rates, and the D2 verdict. Pure python + numpy."""
import json
import numpy as np

SEEDS = (1006, 1007)
WIN = 50                      # form+50 window, aligned per seed at its own formation step
STAT = "c_mem_x"              # the trigger candidate (max cosine to a different fact within the 25-step memory)
PASS_AUROC, CLOSE_AUROC = 0.85, 0.75     # CONTRACT_DISPERSE D3 threshold kept; close band added (PLAN_FINAL §3)
N_PERM = 10000
FPRS = (0.01, 0.001)
EXPECTED_FORMATION = {1006: 180, 1007: 185}   # DISPERSE_READOUT.json; a deterministic re-run must reproduce them


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def formation(curve):
    """First step of two consecutive curve points with trigger_hits >= 12 and trigger - clean_target >= 9
    (disperse_analyze.formation, verbatim rule)."""
    return next((a["step"] for a, b in zip(curve, curve[1:])
                 if all(p["trigger_hits"] >= 12 and p["trigger_hits"] - p["clean_target_hits"] >= 9 for p in (a, b))), None)


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(allv, kind="mergesort"); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    vals, inv = np.unique(allv, return_inverse=True)
    if len(vals) < len(allv):
        sums = np.bincount(inv, weights=ranks); cnts = np.bincount(inv); ranks = (sums / cnts)[inv]
    r = ranks[lab == 1].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def window_rows(rows, lo, hi, stat=STAT):
    return [r for r in rows if lo <= r["step"] <= hi and r.get(stat) is not None]


def perm_null(pos, neg, n_perm=N_PERM, seed=0):
    """AUROC distribution under random poison labels within the window (same n_pos)."""
    rng = np.random.default_rng(seed)
    allv = np.asarray(list(pos) + list(neg), float); k = len(pos); out = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(allv); out[i] = auroc(allv[:k], allv[k:])
    return {"q975": float(np.quantile(out, 0.975)), "q95": float(np.quantile(out, 0.95)), "mean": float(out.mean()), "n_perm": n_perm}


def recall_at_fpr(pos, neg, fpr):
    neg = np.sort(np.asarray(neg, float)); k = int(np.ceil((1 - fpr) * len(neg))) - 1
    th = neg[min(max(k, 0), len(neg) - 1)]
    return float(np.mean(np.asarray(pos) > th)), float(th)


def seed_readout(rows_a1, rows_a0_same, rows_a0_other, f, stat=STAT):
    """All numbers for one seed. rows_*: lists of grad_trigger rows. f: formation step."""
    out = {"formation": f}
    if f is None:
        return dict(out, tier="cannot measure (no formation)")
    for wname, (lo, hi) in {"pre": (max(1, f - WIN), f - 1), "form+50": (f, f + WIN)}.items():
        a1 = window_rows(rows_a1, lo, hi, stat)
        pos = [r[stat] for r in a1 if r["poison"]]; neg_in = [r[stat] for r in a1 if not r["poison"]]
        neg_same = [r[stat] for r in window_rows(rows_a0_same, lo, hi, stat)]
        neg_other = [r[stat] for r in window_rows(rows_a0_other, lo, hi, stat)]
        w = {"lo": lo, "hi": hi, "n_pos": len(pos), "n_neg_in": len(neg_in), "n_neg_a0_same": len(neg_same), "n_neg_a0_other": len(neg_other),
             "auroc_in": auroc(pos, neg_in), "auroc_a0_same": auroc(pos, neg_same), "auroc_a0_other": auroc(pos, neg_other)}
        if pos and neg_in:
            w["perm_in"] = perm_null(pos, neg_in)
            for fpr in FPRS:
                rc, th = recall_at_fpr(pos, neg_same if neg_same else neg_in, fpr)
                w[f"recall_fpr{fpr}"] = rc; w[f"th_fpr{fpr}"] = th
        out[wname] = w
    out["tier"] = d2_tier(out["form+50"])
    return out


def d2_tier(w):
    a = w.get("auroc_in"); q = w.get("perm_in", {}).get("q975")
    if a is None or w.get("n_pos", 0) < 3:
        return "cannot measure"
    if a >= PASS_AUROC and (q is None or a > q):
        return "pass"
    if a <= CLOSE_AUROC:
        return "close"
    return "indeterminate"


def overall(tiers):
    t = [tiers[s] for s in SEEDS]
    if any(x == "cannot measure" for x in t):
        return "cannot measure"
    if all(x == "pass" for x in t):
        return "trigger alive in the k4 formation window -> write the detect-and-reject contract"
    if all(x == "close" for x in t):
        return "trigger dead in the k4 formation window -> A line closed (scientific failure)"
    return "indeterminate -> add two seeds before any verdict"
