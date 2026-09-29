"""Frozen measurement + criteria for supplement/jump (see CONTRACT.md).

Pure functions only; rehearsal.py feeds them synthetic data, run.py real data.
Part A: where the ASR jump sits on the poison-exposure axis (data order).
Part B: whether the first-token logit itself jumps where ASR jumps.
"""
import math

import numpy as np

N_ROWS, BATCH, TOTAL, WARM = 20000, 16, 1250, 38   # WARM = ceil(0.03 * 1250)
LOG_EVERY = 20
N_BOOT, BOOT_SEED = 10000, 0


# ---------------------------------------------------------------- Part A
def lam(t):
    """HF cosine-with-warmup multiplier after t scheduler steps."""
    if t < WARM:
        return t / WARM
    prog = (t - WARM) / (TOTAL - WARM)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))


def lr_check(logged):
    """G0: logged (step, lr) pairs against lam(step) * 1e-4. Returns max rel err."""
    errs = [abs(lr - 1e-4 * lam(s)) / max(1e-12, 1e-4 * lam(s)) for s, lr in logged]
    return max(errs)


def row_steps(order, rows):
    """Optimizer step (1-based) at which each row is consumed."""
    pos = np.empty(N_ROWS, dtype=np.int64)
    pos[order] = np.arange(N_ROWS)
    return pos[np.asarray(rows)] // BATCH + 1


def window_counts(steps, n_win):
    """Poison rows per logging window w = 1..n_win (steps 20(w-1)+1 .. 20w)."""
    w = (np.asarray(steps) - 1) // LOG_EVERY
    return np.bincount(w[w < n_win], minlength=n_win)[:n_win]


def exposure(steps, s):
    """K = rows consumed through step s; D = the same weighted by the lr multiplier
    their update used (update at step k uses lam(k-1)); units: full-lr rows."""
    st = np.asarray(steps)
    seen = st[st <= s]
    return int(seen.size), float(sum(lam(k - 1) for k in seen))


def g1(dloss, order, rows, null_orders, n_win):
    """Order gate. r = Pearson(dloss_w, n_w) over the first n_win windows; null =
    the same r under unrelated orders. p = (1 + #null >= r) / (1 + n_null)."""
    def r_of(o):
        n = window_counts(row_steps(o, rows), n_win).astype(float)
        if n.std() == 0:
            return 0.0
        return float(np.corrcoef(dloss[:n_win], n)[0, 1])
    r = r_of(order)
    null = np.array([r_of(o) for o in null_orders])
    p = (1 + int((null >= r).sum())) / (1 + len(null))
    return {"r": r, "p": p, "null_q99": float(np.quantile(null, 0.99)),
            "pass": p <= 0.01}


def first_at(curve, thr):
    """curve: sorted [(step, asr)]. First step with asr >= thr, else None."""
    for s, a in curve:
        if a >= thr:
            return s
    return None


def a1_tier(t_s1, t_s2, k_s1, k_s2):
    """Which axis the two seeds line up on at t50."""
    rg = lambda a, b: abs(a - b) / ((a + b) / 2)
    gs, gk = rg(t_s1, t_s2), rg(k_s1, k_s2)
    if gk <= 0.5 * gs:
        tier = "exposure-aligned"
    elif gs <= 0.5 * gk:
        tier = "step-aligned"
    else:
        tier = "unclear"
    return {"relgap_step": gs, "relgap_exposure": gk, "tier": tier}


def a2_tier(x, lo, hi):
    """Trajectory exposure at t50 against the dose-line bracket [lo, hi]."""
    if lo <= x <= hi:
        return "inside"
    if lo / 1.5 <= x <= hi * 1.5:
        return "near"
    return "outside"


def tau_b(x, asr, tie=0.10):
    """Kendall tau-b between exposure x and ASR; ASR pairs closer than `tie` are ties."""
    n, conc, disc, tx, ty = len(x), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = asr[i] - asr[j]
            dy = 0.0 if abs(dy) < tie else dy
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else float("nan")


def a3_tier(t):
    if t != t:
        return "unclear"
    if t >= 0.999:
        return "same order"
    if t >= 0.6:
        return "mostly same order"
    return "different order"


# ---------------------------------------------------------------- Part B
def asr60(flips):
    return float(np.mean(flips))


def window(steps, asr):
    """b = first imaged step with asr60 >= 0.95; a = last step before b with
    asr60 <= 0.05. None if either is missing."""
    b = next((s for s, v in zip(steps, asr) if v >= 0.95), None)
    if b is None:
        return None
    pre = [s for s, v in zip(steps, asr) if s < b and v <= 0.05]
    return (pre[-1], b) if pre else None


def _at(steps, s):
    """Nearest imaged step at or below s."""
    c = [x for x in steps if x <= s]
    return c[-1] if c else steps[0]


def b_measure(steps, T3, a, b, noise, rng=None):
    """steps: sorted imaged steps; T3: array [n_steps, n_samples] (same samples).
    R = slope of median T3 inside [a, b] / slope over the equal-length window
    before a. C = share of the start->end change of median T3 inside [a, b] over
    the share of steps. CI: bootstrap over samples."""
    steps = list(steps)
    L = b - a
    p = _at(steps, a - L)
    ia, ib, ip = steps.index(a), steps.index(b), steps.index(p)
    i0, i1 = 0, len(steps) - 1

    def stats(M):
        s_in = (M[ib] - M[ia]) / (b - a)
        s_pre = (M[ia] - M[ip]) / max(1, a - p)
        R = s_in / s_pre if s_pre > 0 else float("inf")
        tot = M[i1] - M[i0]
        C = ((M[ib] - M[ia]) / tot) / ((b - a) / (steps[i1] - steps[i0])) if tot > 0 else float("nan")
        return R, C, M[ib] - M[ia]

    M = np.median(T3, axis=1)
    R, C, inside = stats(M)
    rng = rng or np.random.default_rng(BOOT_SEED)
    n = T3.shape[1]
    bs = np.array([stats(np.median(T3[:, rng.integers(0, n, n)], axis=1))[:2]
                   for _ in range(N_BOOT)])
    Rq = np.quantile(bs[:, 0], [0.025, 0.975], method="inverted_cdf")
    Cq = np.nanquantile(bs[:, 1], [0.025, 0.975], method="inverted_cdf")
    valid = inside >= 3 * noise
    if not valid:
        tier = "unclear (inside change < 3x noise)"
    elif Rq[1] <= 1.5:
        tier = "threshold artefact (no acceleration)"
    elif Rq[0] >= 3:
        tier = "logit accelerates"
    else:
        tier = "unclear"
    return {"a": a, "b": b, "pre_start": p, "R": R, "R_ci": Rq.tolist(),
            "C": C, "C_ci": Cq.tolist(), "inside_change": float(inside),
            "noise": noise, "tier": tier}


def noise_of(ref_medians):
    """Median |change| of a clean reference's median T3 between consecutive refs."""
    d = np.abs(np.diff(np.asarray(ref_medians)))
    return float(np.median(d)) if d.size else float("nan")
