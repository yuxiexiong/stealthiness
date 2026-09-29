"""Round 2, part G: blind prediction of grok10 take-off steps from data order alone.

Pure functions; rehearsal2.py feeds synthetic data, run_grok_pred.py real data.
Frozen with CONTRACT2.md.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jump as J  # noqa: E402

K5, K50 = 48, 53          # from round 1: s1/s2 poison count at t5_20 (48, 48) and t50_20 (54, 52)
MIN_SEEDS = 6
RATIO_STRONG = 0.7


def predict(order, rows, K):
    """Exact optimizer step at which the K-th poison row is consumed."""
    return int(np.sort(J.row_steps(order, rows))[K - 1])


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(pred, act):
    """One-sided exact permutation p of Spearman rho (all n! assignments)."""
    pred, act = np.asarray(pred, float), np.asarray(act, float)
    r = spearman(pred, act)
    null = np.array([spearman(pred[list(p)], act)
                     for p in itertools.permutations(range(len(pred)))])
    return r, float((null >= r - 1e-12).mean())


def every20(curve):
    return [(s, a) for s, a in sorted(curve) if s % 20 == 0]


def g_tier(pred, act):
    """pred: exact predicted steps; act: observed every-20 steps (None = missing)."""
    ok = [(p, a) for p, a in zip(pred, act) if a is not None]
    if len(ok) < MIN_SEEDS:
        return {"n": len(ok), "tier": "cannot measure (fewer than %d seeds)" % MIN_SEEDS}
    p_, a_ = np.array([x[0] for x in ok], float), np.array([x[1] for x in ok], float)
    rho, p = perm_p(p_, a_)
    mae_k = float(np.mean(np.abs(p_ - a_)))
    mae_c = float(np.mean(np.abs(np.median(a_) - a_)))   # oracle constant: favours the step-axis null
    ratio = mae_k / mae_c if mae_c > 0 else float("inf")
    if p <= 0.05 and ratio <= RATIO_STRONG:
        tier = "order sets the take-off step"
    elif p <= 0.05:
        tier = "order matters, prediction error large"
    elif rho <= 0:
        tier = "no order effect seen"
    else:
        tier = "unclear"
    return {"n": len(ok), "rho": rho, "p": p, "mae_K": mae_k, "mae_const": mae_c,
            "ratio": ratio, "tier": tier}
