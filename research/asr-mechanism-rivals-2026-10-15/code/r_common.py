"""Frozen rules of CONTRACT_R.md (rival-set stage R). numpy only; no model, no GPU.

A trajectory is (steps, X, F): steps sorted ints, X (points x inputs) per-input continuous quantity
(target-vs-reference margin), F (points x inputs) flip indicator. c(t) = mean over inputs of X.
"""
import math

import numpy as np

LOW, HIGH = 0.10, 0.90
# --- stage R tier thresholds (CONTRACT_R.md section 4) ---
TRANSIENT_MAX = 1.5        # K1 median A_half at or below this: acceleration gone
INTRINSIC_KEEP = 0.7       # K1 median A_half >= 0.7 x K0 median: acceleration kept
DENS_HI, DENS_LO = 0.7, 0.3
CLOCK_TOL = 0.25           # relative cumulative-LR agreement
STEP_TOL = 10              # updates
DELAY_MIN = 10             # updates
ACC_CLEAN = 0.80
ADD_OK = 0.80
RANK_HI, RANK_LO = 0.6, 0.2
LEAK_HI, LEAK_LO = 0.5, 0.1
SCHED_TOL = 0.5            # |log2(observed ratio / LR ratio)|


# ---------------------------------------------------------------- window and speeds
def window(steps, asr):
    """Atlas window: b = first point with ASR >= 0.90; a = last point before b with ASR <= 0.10."""
    b = next((s for s, v in zip(steps, asr) if v >= HIGH), None)
    if b is None:
        return None
    pre = [s for s, v in zip(steps, asr) if s < b and v <= LOW]
    return (pre[-1], b) if pre else None


def first_at_least(steps, series, thr):
    return next((s for s, v in zip(steps, series) if v >= thr), None)


def atlas_ratio(steps, c, a, b):
    """Atlas M2 ratio v_in / v_pre with p = nearest point <= a - (b - a). None if p would precede the data."""
    steps = list(steps)
    ptar = a - (b - a)
    if ptar < steps[0]:
        return None
    p = max(s for s in steps if s <= ptar)
    ip, ia, ib = steps.index(p), steps.index(a), steps.index(b)
    v_pre = (c[ia] - c[ip]) / max(1, a - p)
    v_in = (c[ib] - c[ia]) / (b - a)
    return {"p": p, "v_pre": float(v_pre), "v_in": float(v_in),
            "ratio": float(v_in / v_pre) if v_pre > 0 else math.inf}


def nearest_le(steps, t):
    return max(s for s in steps if s <= t)


def a_half(steps, c, b):
    """Primary acceleration measure of stage R: mean speed of c over [b/2, b] divided by mean speed over
    [s0, b/2], where s0 is the first point. Defined when b - s0 >= 4 points of separation."""
    steps = list(steps)
    s0 = steps[0]
    if b is None or b - s0 < 4:
        return None
    h = nearest_le(steps, s0 + (b - s0) / 2)
    if h <= s0 or h >= b:
        return None
    i0, ih, ib = 0, steps.index(h), steps.index(b)
    v1 = (c[ih] - c[i0]) / (h - s0)
    v2 = (c[ib] - c[ih]) / (b - h)
    return float(v2 / v1) if v1 > 0 else math.inf


def v_mean(steps, c, b):
    steps = list(steps)
    if b is None or b <= steps[0]:
        return None
    return float((c[steps.index(b)] - c[0]) / (b - steps[0]))


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return None
    lx, ly = np.log2(x[ok]), np.log2(y[ok])
    if np.ptp(lx) == 0:
        return None
    return float(np.polyfit(lx, ly, 1)[0])


def cum_lr(lrs, t):
    """Sum of the learning rates of updates 1..t (lrs[k] = LR used by update k+1)."""
    return float(np.sum(lrs[:t]))


def summarize_run(steps, X, F, acc=None):
    """Per-run quantities used by every stage-R tier."""
    steps = list(steps)
    X, F = np.asarray(X, float), np.asarray(F, bool)
    asr = F.mean(1)
    c = X.mean(1)
    t50 = first_at_least(steps, asr, 0.5)
    t90 = first_at_least(steps, asr, HIGH)
    w = window(steps, asr)
    out = {"n_points": len(steps), "asr_first": float(asr[0]), "asr_max": float(asr.max()),
           "t50": t50, "t90": t90, "window": list(w) if w else None,
           "A_half": a_half(steps, c, t90), "v_mean": v_mean(steps, c, t90)}
    out["atlas"] = atlas_ratio(steps, c, *w) if w else None
    if acc is not None:
        out["t_clean80"] = first_at_least(steps, acc, ACC_CLEAN)
        out["acc_first"] = float(acc[0])
        out["acc_at_t90"] = float(acc[steps.index(t90)]) if t90 is not None else None
    return out


# ---------------------------------------------------------------- band-width (group A) quantities
def rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / counts)[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def additive_share(Xs):
    """Share of the variance of X (after removing the grand mean) explained by c(t) + o_i."""
    Xs = np.asarray(Xs, float)
    g = Xs.mean()
    fit = Xs.mean(1, keepdims=True) + Xs.mean(0, keepdims=True) - g
    tot = ((Xs - g) ** 2).sum()
    return float(1 - ((Xs - fit) ** 2).sum() / tot) if tot > 0 else None


def offsets(Xs):
    """o_i = mean over the slice of (x_i - c): the input's position in the band (higher = flips earlier)."""
    Xs = np.asarray(Xs, float)
    return Xs.mean(0) - Xs.mean()


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def corpus_run(steps, X, F, ids, Xclean=None, lrs=None):
    """Group-A and schedule-null quantities for one existing corpus run (CONTRACT_R.md section 3.1)."""
    steps = list(steps)
    X, F = np.asarray(X, float), np.asarray(F, bool)
    asr = F.mean(1)
    w = window(steps, asr)
    rec = {"window": list(w) if w else None}
    if not w:
        rec["status"] = "no_window"
        return rec
    a, b = w
    ar = atlas_ratio(steps, X.mean(1), a, b)
    p = ar["p"] if ar else steps[0]
    ip, ib = steps.index(p), steps.index(b)
    Xs = X[ip:ib + 1]
    o = offsets(Xs)
    rec.update(status="ok", p=p, additive_share=additive_share(Xs),
               inherit_rho=spearman(o, X[0]), offsets=dict(zip(map(str, ids), map(float, o))),
               atlas_ratio=ar["ratio"] if ar else None, first_step=steps[0])
    if Xclean is not None:
        cc = np.asarray(Xclean, float).mean(1)
        rec["leak_slope"] = ols_slope(X.mean(1)[ip:ib + 1], cc[ip:ib + 1])
    if lrs is not None and ar:
        lr = np.asarray(lrs, float)
        pre = lr[p:a].mean() if a > p else None       # updates p+1..a
        inw = lr[a:b].mean()                           # updates a+1..b
        lrr = float(inw / pre) if pre else None
        rec["lr_ratio"] = lrr
        rec["sched_log2"] = (float(np.log2(ar["ratio"] / lrr)) if (lrr and np.isfinite(ar["ratio"]) and ar["ratio"] > 0)
                             else None)
    return rec


def cross_seed_rho(recs):
    """Median pairwise Spearman of offsets over runs of one family on their common probe ids."""
    ok = [r for r in recs if r.get("status") == "ok"]
    vals = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            common = sorted(set(ok[i]["offsets"]) & set(ok[j]["offsets"]))
            if len(common) >= 20:
                v = spearman([ok[i]["offsets"][k] for k in common], [ok[j]["offsets"][k] for k in common])
                if v is not None:
                    vals.append(v)
    return (float(np.median(vals)) if vals else None), len(vals)


# ---------------------------------------------------------------- tiers
def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def tier_transient(k0, k1):
    """B0' fresh-optimizer transient vs intrinsic acceleration (per model, medians over seeds)."""
    m0, m1 = med([r["A_half"] for r in k0]), med([r["A_half"] for r in k1])
    if m0 is None or m1 is None:
        return "unmeasurable", m0, m1
    if m1 <= TRANSIENT_MAX and m0 >= 2 * m1:
        return "transient", m0, m1
    if m1 >= INTRINSIC_KEEP * m0:
        return "intrinsic", m0, m1
    return "partial", m0, m1


def tier_density(rho, runs_by_rho):
    """B4 arrival-rate kinematics: exponents of v_mean and t90 on poison density (medians over seeds)."""
    vs = [med([r["v_mean"] for r in runs_by_rho[k]]) for k in rho]
    ts = [med([r["t90"] for r in runs_by_rho[k]]) for k in rho]
    ev, et = loglog_slope(rho, vs), loglog_slope(rho, ts)
    if ev is None:
        return "unmeasurable", ev, et
    if ev >= DENS_HI and (et is None or et <= -DENS_HI):
        return "arrival_rate", ev, et
    if abs(ev) <= DENS_LO:
        return "density_insensitive", ev, et
    return "partial", ev, et


def tier_shortcut(k0, k2):
    """C3 gating vs C5 shortcut on the no-adaptation start (per seed, then majority)."""
    labels = []
    for r0, r2 in zip(k0, k2):
        t90, tc = r2.get("t90"), r2.get("t_clean80")
        if t90 is None:
            labels.append("no_install")
        elif tc is None or t90 < tc:
            labels.append("shortcut")
        elif r0.get("t90") is not None and t90 - r0["t90"] >= DELAY_MIN:
            labels.append("gated")
        else:
            labels.append("mixed")
    return majority(labels), labels


def tier_clock(l0, l1, lr0, lr1):
    """C0 onset clock: cumulative LR vs update count (per seed, then majority)."""
    labels = []
    for r0, r1, a, b in zip(l0, l1, lr0, lr1):
        t0, t1 = r0.get("t90"), r1.get("t90")
        if t0 is None or t1 is None:
            labels.append("unmeasurable")
            continue
        c0, c1 = cum_lr(a, t0), cum_lr(b, t1)
        if abs(c1 / c0 - 1) <= CLOCK_TOL and t1 <= t0 - STEP_TOL:
            labels.append("lr_clock")
        elif abs(t1 - t0) <= STEP_TOL:
            labels.append("step_clock")
        else:
            labels.append("mixed")
    return majority(labels), labels


def majority(labels):
    if not labels:
        return "none"
    u, n = np.unique(labels, return_counts=True)
    top = n.max()
    win = [x for x, k in zip(u, n) if k == top]
    return str(win[0]) if len(win) == 1 else "mixed"


def tier_family_A(recs, rho_cross):
    ok = [r for r in recs if r.get("status") == "ok"]
    add = med([r["additive_share"] for r in ok])
    inh = med([r["inherit_rho"] for r in ok])
    leak = med([r.get("leak_slope") for r in ok])
    t = {"additive_share": add, "inherit_rho": inh, "cross_seed_rho": rho_cross, "leak_slope": leak}
    t["A1"] = None if add is None else ("deleted" if add >= ADD_OK else "alive")
    t["stability"] = None if rho_cross is None else ("stable" if rho_cross >= RANK_HI else
                                                      ("seed_specific" if rho_cross <= RANK_LO else "middling"))
    t["inheritance"] = None if inh is None else ("inherited" if inh >= RANK_HI else
                                                  ("formed" if inh <= RANK_LO else "middling"))
    t["leak"] = None if leak is None else ("leak" if leak >= LEAK_HI else ("specific" if leak <= LEAK_LO else "partial"))
    return t


def tier_schedule(recs):
    vals = [r.get("sched_log2") for r in recs if r.get("status") == "ok"]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "unmeasurable", None
    expl = sum(abs(v) <= SCHED_TOL for v in vals)
    return ("schedule_explained" if expl > len(vals) / 2 else "not_schedule"), f"{expl}/{len(vals)}"
