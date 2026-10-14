"""Frozen rules for the Q1 experiment (CONTRACT_Q1.md): windows, the r2 decomposition (D1/D2/D3, ported from
attribution-microscope/supplement/jump/r2/decomp.py and jump.py without changing the statistics), the threshold
sweep, the breakpoint fit and the judge. numpy only; nothing here touches the model, the server or the GPU.

Conventions: X is a (steps x rows) matrix of first-token margins logit(target) - logit(gold) under the trigger
condition; F is the same-shaped flip indicator (first-token argmax == target). `steps` is the sorted list of
updates that index the rows of X.
"""
import numpy as np

HORIZON = 160
N_ROWS = 60
AGREE_MIN = 0.90                 # D gate (r2): flips must agree with the sign of the margin on >= 90% of cells
LOW, HIGH = 0.10, 0.90           # primary window: ASR60 from <= 0.10 (6 hits) to >= 0.90 (54 hits)
LOW_R2, HIGH_R2 = 0.05, 0.95     # secondary window: the r2 rule verbatim (asr60 <= 0.05 -> >= 0.95)
ACCEL_MIN = 3.0                  # real jump: common course >= 3x faster inside the window than before it
ACCEL_ARTIFACT = 1.5             # threshold artifact: < 1.5x and ...
D3_MIN = 0.8                     # real jump: >= 80% of rows accelerate on their own (k = 1.5)
D3_ARTIFACT = 0.5                # ... < 50% of rows accelerate
D3_K = 1.5
TAUS = tuple(range(-8, 9, 2))    # threshold sweep
TAU_BAND = 4                     # widths for |tau| <= 4 are compared with the width at tau = 0
WIDTH_FACTOR = 2.0
POST = 20                        # breakpoint fit on [0, min(HORIZON, b + POST)]
D1_LO, D1_HI = 0.15, 0.25        # r2 D1 bands (descriptive here)
N_BOOT, BOOT_SEED = 10000, 0


# --------------------------------------------------------------------------- windows

def window(steps, asr, low, high):
    """b = first step with asr >= high; a = last step before b with asr <= low. None if either is missing.
    (jump.window with the thresholds as parameters; r2 used 0.05 / 0.95.)"""
    b = next((s for s, v in zip(steps, asr) if v >= high), None)
    if b is None:
        return None
    pre = [s for s, v in zip(steps, asr) if s < b and v <= low]
    return (pre[-1], b) if pre else None


def at(steps, s):
    """Nearest available step at or below s (jump._at)."""
    c = [x for x in steps if x <= s]
    return c[-1] if c else steps[0]


def span(steps, asr, low, high):
    """(p, a, b): the window [a, b] and the equal-length pre-window start p = a - (b - a), clipped to the data."""
    w = window(steps, asr, low, high)
    if w is None:
        return None
    a, b = w
    return at(list(steps), a - (b - a)), a, b


# --------------------------------------------------------------------------- r2 decomposition (verbatim statistics)

def additive(X):
    """Two-way additive fit X_is ~ row_s + col_i (steps x samples)."""
    g = X.mean()
    return X.mean(1, keepdims=True) + X.mean(0, keepdims=True) - g


def d1_stat(X, F):
    return float(np.max(np.abs((additive(X) > 0).mean(1) - F.mean(1))))


def d1(X, F, rng=None, n_boot=N_BOOT):
    """Gate + descriptive additive check. X, F restricted to the span [p, b]."""
    agree = float(((X > 0) == F).mean())
    if agree < AGREE_MIN:
        return {"agree": agree, "tier": "cannot measure (flip not set by margin sign)"}
    D = d1_stat(X, F)
    rng = rng or np.random.default_rng(BOOT_SEED)
    n = X.shape[1]
    bs = []
    for _ in range(n_boot):
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
    return {"agree": agree, "D": D, "D_ci": [float(lo), float(hi)], "R2_additive": r2, "tier": tier}


def d2(steps_span, X, p, a, b):
    """Sample i flips when the common course c_s passes theta_i. Speeds of c before / inside the window."""
    c = X.mean(1)
    theta = X.mean() - X.mean(0)
    st = list(steps_span)
    ip, ia, ib = st.index(p), st.index(a), st.index(b)
    v_in = (c[ib] - c[ia]) / (b - a)
    v_pre = (c[ia] - c[ip]) / max(1, a - p)
    W = float(np.quantile(theta, 0.95) - np.quantile(theta, 0.05))
    return {"W_5to95": W, "v_in": float(v_in), "v_pre": float(v_pre),
            "accel_ratio": float(v_in / v_pre) if v_pre > 0 else float("inf"),
            "steps_5to95_at_v_in": W / v_in if v_in > 0 else float("inf"),
            "steps_5to95_at_v_pre": W / v_pre if v_pre > 0 else float("inf"),
            "theta_q": np.quantile(theta, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
            "course": c.tolist()}


def d3(steps_span, X, p, a, b, k=D3_K):
    """Share of samples whose own slope in [a, b] exceeds k x their own slope in [p, a]."""
    st = list(steps_span)
    ip, ia, ib = st.index(p), st.index(a), st.index(b)
    s_in = (X[ib] - X[ia]) / (b - a)
    s_pre = (X[ia] - X[ip]) / max(1, a - p)
    acc = np.where(s_pre > 0, s_in > k * s_pre, s_in > 0)
    return {"share_accelerating": float(acc.mean()), "share_pre_slope_nonpos": float((s_pre <= 0).mean())}


# --------------------------------------------------------------------------- threshold sweep

def threshold_sweep(steps, X, taus=TAUS, low=LOW, high=HIGH):
    """ASR_tau(t) = share of rows with margin > tau; width_tau = b - a of the primary window on that curve."""
    out = {}
    for tau in taus:
        asr = (X > tau).mean(1)
        w = window(steps, asr, low, high)
        out[str(tau)] = {"window": list(w) if w else None, "width": (w[1] - w[0]) if w else None}
    return out


def sweep_equal_widths(sweep, band=TAU_BAND, factor=WIDTH_FACTOR):
    """Descriptive. True when every width for |tau| <= band exists and lies within `factor` of the width at tau = 0:
    the signature of a constant-speed common course (ASR_tau curves are time-shifted copies). Under a course that
    accelerates at some margin level, widths depend on tau and this returns False."""
    w0 = sweep["0"]["width"]
    if w0 is None:
        return False
    for tau, v in sweep.items():
        if abs(int(tau)) <= band:
            if v["width"] is None or not (w0 / factor <= v["width"] <= w0 * factor):
                return False
    return True


# --------------------------------------------------------------------------- breakpoint fit on the common course

def _bic(rss, n, k):
    return n * np.log(max(rss, 1e-12) / n) + k * np.log(n)


def breakpoint_fit(t, c):
    """Linear vs quadratic vs one-breakpoint piecewise-linear (continuous) fits of c(t). Returns BICs and the break."""
    t, c = np.asarray(t, float), np.asarray(c, float)
    n = len(t)
    A1 = np.vstack([np.ones(n), t]).T
    rss1 = float(np.sum((c - A1 @ np.linalg.lstsq(A1, c, rcond=None)[0]) ** 2))
    A2 = np.vstack([np.ones(n), t, t ** 2]).T
    rss2 = float(np.sum((c - A2 @ np.linalg.lstsq(A2, c, rcond=None)[0]) ** 2))
    best = None
    for i in range(2, n - 2):
        tb = t[i]
        A3 = np.vstack([np.ones(n), t, np.maximum(t - tb, 0)]).T
        coef = np.linalg.lstsq(A3, c, rcond=None)[0]
        rss3 = float(np.sum((c - A3 @ coef) ** 2))
        if best is None or rss3 < best[0]:
            best = (rss3, float(tb), coef.tolist())
    rss3, tb, coef = best
    return {"bic_linear": _bic(rss1, n, 2), "bic_quadratic": _bic(rss2, n, 3), "bic_breakpoint": _bic(rss3, n, 4),
            "delta_bic_linear_minus_break": _bic(rss1, n, 2) - _bic(rss3, n, 4),
            "delta_bic_quadratic_minus_break": _bic(rss2, n, 3) - _bic(rss3, n, 4),
            "break_step": tb, "slope_before": coef[1], "slope_after": coef[1] + coef[2]}


# --------------------------------------------------------------------------- per-mother analysis and judge

def analyze_mother(steps, X, F, low=LOW, high=HIGH, n_boot=N_BOOT):
    """Everything the contract asks for on one mother and one window rule."""
    steps = list(steps)
    asr = F.mean(1)
    sp = span(steps, asr, low, high)
    out = {"asr": asr.tolist(), "window_rule": [low, high], "span": None}
    if sp is None:
        out["measurable"] = False
        out["reason"] = "no window under this rule"
        return out
    p, a, b = sp
    ip, ib = steps.index(p), steps.index(b)
    Xs, Fs, ss = X[ip:ib + 1], F[ip:ib + 1], steps[ip:ib + 1]
    out.update(span=[p, a, b], d1=d1(Xs, Fs, n_boot=n_boot))
    out["measurable"] = "D" in out["d1"]
    if not out["measurable"]:
        out["reason"] = out["d1"]["tier"]
        return out
    out["d2"] = d2(ss, Xs, p, a, b)
    out["d3"] = d3(ss, Xs, p, a, b)
    out["sweep"] = threshold_sweep(steps, X, low=low, high=high)
    out["sweep_equal_widths"] = sweep_equal_widths(out["sweep"])
    stop = min(HORIZON, b + POST)
    ie = steps.index(at(steps, stop))
    out["breakpoint"] = breakpoint_fit(steps[:ie + 1], X[:ie + 1].mean(1))
    return out


def classify_mother(m):
    """real / artifact / between / unmeasurable for one mother under the primary rule."""
    if not m.get("measurable"):
        return "unmeasurable"
    r, s = m["d2"]["accel_ratio"], m["d3"]["share_accelerating"]
    if r >= ACCEL_MIN and s >= D3_MIN:
        return "real"
    if r < ACCEL_ARTIFACT and s < D3_ARTIFACT:
        return "artifact"
    return "between"


def judge(classes):
    """classes: {mother: class}. Frozen tiers."""
    vals = list(classes.values())
    if vals and all(v == "real" for v in vals):
        tier = "real_jump"
    elif any(v == "artifact" for v in vals):
        tier = "threshold_artifact"
    else:
        tier = "mixed"
    return {"tier": tier, "per_mother": classes}
