"""Round 2, part D: per-sample decomposition of the ASR jump (60 imaged samples).

Question: is the jump one common rise of T3 that sweeps past fixed per-sample
offsets (then ASR steepness = spread of offsets / speed of the common rise), or
do samples follow their own courses? Frozen with CONTRACT2.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jump as J  # noqa: E402

AGREE_MIN = 0.90
D1_LO, D1_HI = 0.15, 0.25
N_BOOT, BOOT_SEED = 10000, 0


def span(steps, asr):
    """[p, b] = B's pre-window plus window. None if no window."""
    w = J.window(steps, asr)
    if w is None:
        return None
    a, b = w
    return J._at(list(steps), a - (b - a)), a, b


def additive(X):
    """Two-way additive fit X_is ~ row_s + col_i (steps x samples)."""
    g = X.mean()
    return X.mean(1, keepdims=True) + X.mean(0, keepdims=True) - g


def d1_stat(X, F):
    return float(np.max(np.abs((additive(X) > 0).mean(1) - F.mean(1))))


def d1(X, F, rng=None):
    """X: T3 in span (steps x samples); F: flip indicator (argmax == target)."""
    agree = float(((X > 0) == F).mean())
    if agree < AGREE_MIN:
        return {"agree": agree, "tier": "cannot measure (flip not set by T3 sign)"}
    D = d1_stat(X, F)
    rng = rng or np.random.default_rng(BOOT_SEED)
    n = X.shape[1]
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        bs.append(d1_stat(X[:, i], F[:, i]))
    lo, hi = np.quantile(bs, [0.025, 0.975], method="inverted_cdf")
    res = X - additive(X)
    r2 = 1 - float((res ** 2).sum() / ((X - X.mean()) ** 2).sum())
    if hi <= D1_LO + 1e-9:
        tier = "common shift"
    elif lo >= D1_HI - 1e-9:
        tier = "samples go their own way"
    else:
        tier = "unclear"
    return {"agree": agree, "D": D, "D_ci": [float(lo), float(hi)], "R2_additive": r2,
            "tier": tier}


def d2(steps_span, X, p, a, b):
    """Descriptive. Sample i flips when common course c_s passes theta_i."""
    c = X.mean(1)
    theta = X.mean() - X.mean(0)                   # fit c_s + col_i - g > 0  <=>  c_s > g - col_i
    st = list(steps_span)
    ip, ia, ib = st.index(p), st.index(a), st.index(b)
    v_in = (c[ib] - c[ia]) / (b - a)
    v_pre = (c[ia] - c[ip]) / max(1, a - p)
    W = float(np.quantile(theta, 0.95) - np.quantile(theta, 0.05))
    return {"W_5to95": W, "v_in": float(v_in), "v_pre": float(v_pre),
            "steps_5to95_at_v_in": W / v_in if v_in > 0 else float("inf"),
            "steps_5to95_at_v_pre": W / v_pre if v_pre > 0 else float("inf"),
            "theta_q": np.quantile(theta, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
            "course": c.tolist()}


def d3(steps_span, X, p, a, b, k=1.5):
    """Descriptive. Share of samples whose own slope in [a,b] > k x own slope in [p,a]."""
    st = list(steps_span)
    ip, ia, ib = st.index(p), st.index(a), st.index(b)
    s_in = (X[ib] - X[ia]) / (b - a)
    s_pre = (X[ia] - X[ip]) / max(1, a - p)
    acc = np.where(s_pre > 0, s_in > k * s_pre, s_in > 0)
    return {"share_accelerating": float(acc.mean()),
            "share_pre_slope_nonpos": float((s_pre <= 0).mean())}
