"""Frozen rules for the platform-exit experiment (CONTRACT_EXIT.md). numpy only; no model, server or GPU."""
import numpy as np

A = {1105: 48, 1106: 88, 1107: 71}           # Q1 primary window starts
FIRST_CANDIDATE = 14
EXIT_RISE = 2.0                              # descriptive exit: c(u+4) - c(u-1) >= 2.0
# A1
A1_SPAN = 40
A1_RATIO, A1_MIN = 3.0, 1.0
# A2
BURST_MIN = 1.0
SHARE_ROW, SHARE_NOT = 0.8, 0.5
CTRL_MAX = 0.2
# A3
RATIO_UP, RATIO_DOWN = 1.5, 0.67
EDGE = 3
# B1
B1_LOW, B1_HIGH = 0.10, 0.90
B1_LEAD = 20
B1_DBIC = 10.0
B1_SLOPE_FACTOR = 1.5
B1_R_IN = 3.0
B1_UNRES = 10


def candidates(poison_updates, a):
    return sorted(u for u in poison_updates if FIRST_CANDIDATE <= u <= a)


def exit_update(c, poison_updates, a):
    """c: list/array indexed by update (c[0..]). First candidate with c(u+4) - c(u-1) >= EXIT_RISE, else None."""
    for u in candidates(poison_updates, a):
        if u + 4 < len(c) and c[u + 4] - c[u - 1] >= EXIT_RISE:
            return u
    return None


# --------------------------------------------------------------------------- A1

def a1_stats(inc, poison_updates, lo, hi):
    """inc: {update: increment}; returns m_p, m_c, counts over [lo, hi]."""
    p = [v for u, v in inc.items() if lo <= u <= hi and u in poison_updates]
    q = [v for u, v in inc.items() if lo <= u <= hi and u not in poison_updates]
    return {"m_p": float(np.mean(p)) if p else None, "m_c": float(np.mean(q)) if q else None, "n_p": len(p), "n_c": len(q)}


def a1_poison_driven(s):
    return s["m_p"] is not None and s["m_c"] is not None and s["m_p"] >= A1_MIN and s["m_p"] >= A1_RATIO * s["m_c"]


# --------------------------------------------------------------------------- A2

def a2_mother(rows):
    """rows: [{"u", "inc": {variant: value}}]. Class on burst updates (inc_base >= BURST_MIN)."""
    burst = [r for r in rows if r["inc"]["base"] >= BURST_MIN]
    if not burst:
        return {"class": "no_burst", "n_burst": 0}
    share = [1 - r["inc"]["P_to_clean"] / r["inc"]["base"] for r in burst]
    ctrl = [abs(r["inc"]["C_drop"] - r["inc"]["base"]) / r["inc"]["base"] for r in burst]
    drop = [1 - r["inc"]["P_drop"] / r["inc"]["base"] for r in burst]
    ms, mc = float(np.median(share)), float(np.median(ctrl))
    cls = "row_caused" if ms >= SHARE_ROW and mc <= CTRL_MAX else ("not_row_caused" if ms < SHARE_NOT else "between")
    return {"class": cls, "n_burst": len(burst), "burst_updates": [r["u"] for r in burst], "median_share_P": ms,
            "median_ctrl": mc, "median_share_P_drop": float(np.median(drop)), "share_P": share, "ctrl": ctrl}


def a2_tier(classes):
    v = list(classes.values())
    if v and all(x == "row_caused" for x in v):
        return "row_caused"
    if any(x == "not_row_caused" for x in v):
        return "not_row_caused"
    return "mixed"


# --------------------------------------------------------------------------- A3

def edge_ratio(values):
    """mean of the last EDGE values / mean of the first EDGE values (values in state order)."""
    v = np.asarray(values, float)
    first, last = v[:EDGE].mean(), v[-EDGE:].mean()
    if first <= 0:
        return float("inf") if last > 0 else float("nan")
    return float(last / first)


def a3_class(ratio_b, ratio_m, b2_retreats):
    if ratio_b >= RATIO_UP and b2_retreats:
        return "gating"
    if ratio_b <= RATIO_DOWN and ratio_m >= RATIO_UP:
        return "autocatalysis"
    if RATIO_DOWN < ratio_b < RATIO_UP and RATIO_DOWN < ratio_m < RATIO_UP:
        return "flat"
    return "unclear"


def same_or_inconsistent(classes):
    v = set(classes.values())
    return v.pop() if len(v) == 1 else "inconsistent"


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- B1

def window(steps, asr, low=B1_LOW, high=B1_HIGH):
    b = next((s for s, v in zip(steps, asr) if v >= high), None)
    if b is None:
        return None
    pre = [s for s, v in zip(steps, asr) if s < b and v <= low]
    return (pre[-1], b) if pre else None


def _bic(rss, n, k):
    return n * np.log(max(rss, 1e-12) / n) + k * np.log(n)


def breakpoint_fit(t, c):
    t, c = np.asarray(t, float), np.asarray(c, float)
    n = len(t)
    A1m = np.vstack([np.ones(n), t]).T
    rss1 = float(np.sum((c - A1m @ np.linalg.lstsq(A1m, c, rcond=None)[0]) ** 2))
    best = None
    for i in range(2, n - 2):
        A3m = np.vstack([np.ones(n), t, np.maximum(t - t[i], 0)]).T
        coef = np.linalg.lstsq(A3m, c, rcond=None)[0]
        rss = float(np.sum((c - A3m @ coef) ** 2))
        if best is None or rss < best[0]:
            best = (rss, float(t[i]), coef)
    rss3, tb, coef = best
    return {"break": tb, "slope_before": float(coef[1]), "slope_after": float(coef[1] + coef[2]),
            "dbic": float(_bic(rss1, n, 2) - _bic(rss3, n, 4))}


def slope(steps, c, s0, s1):
    i0, i1 = steps.index(s0), steps.index(s1)
    return (c[i1] - c[i0]) / (s1 - s0) if s1 > s0 else float("nan")


def b1_arm(steps, gap, hit):
    """steps: sorted checkpoint steps; gap, hit: (steps x samples). Returns the per-arm record and class."""
    steps = list(steps)
    asr = hit.mean(1)
    c = gap.mean(1)
    w = window(steps, asr)
    out = {"steps": steps, "asr60": asr.tolist(), "course": c.tolist(), "window": list(w) if w else None}
    if w is None:
        out["class"] = "no_window"
        return out
    a, b = w
    p = max(steps[0], a - (b - a))
    p = max(s for s in steps if s <= p) if any(s <= p for s in steps) else steps[0]
    bp = breakpoint_fit(steps, c)
    t = bp["break"]
    v_pre, v_in = slope(steps, c, p, a), slope(steps, c, a, b)
    s_ta = slope(steps, c, int(t), a) if t < a else float("nan")
    a20 = max(s for s in steps if s <= a - B1_LEAD) if any(s <= a - B1_LEAD for s in steps) else steps[0]
    s_last = slope(steps, c, a20, a) if a20 < a else float("nan")
    R = v_in / v_pre if v_pre > 0 else float("inf")
    out.update(pre=p, breakpoint=bp, v_pre=v_pre, v_in=v_in, slope_exit_to_a=s_ta, slope_last20_before_a=s_last, R=R)
    if t <= a - B1_LEAD and bp["dbic"] >= B1_DBIC and v_in <= B1_SLOPE_FACTOR * s_ta:
        cls = "exit_precedes"
    elif R >= B1_R_IN and v_in > B1_SLOPE_FACTOR * s_last:
        cls = "in_window"
    elif a - B1_UNRES <= t <= a:
        cls = "unresolved"
    else:
        cls = "other"
    out["class"] = cls
    return out


def b1_tier(classes):
    v = list(classes.values())
    n = len(v)
    if sum(x == "exit_precedes" for x in v) >= 6:
        return "toy_pattern"
    if sum(x == "in_window" for x in v) >= 6:
        return "llava_accelerates"
    if sum(x == "unresolved" for x in v) >= 4:
        return "resolution_limited"
    return "mixed" if n else "none"
