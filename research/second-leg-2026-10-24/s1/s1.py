"""S1 (CONTRACT_S1): binary-budget reading of frozen ASR readouts. Pure functions, stdlib only.

Nothing here reads a raw margin/LP series. Inputs are the 10-step greedy ASR grid and
clean_acc, as already written by the frozen evaluators.
"""
import json
import math

GRID = 10                      # step grid of the behavioural readouts
HOLD = 40                      # sf.t50_sustained hold window
THR = 0.5                      # hit threshold on greedy ASR
Z_SUM = 1.959963985 + 0.841621234   # z_{0.975} + z_{0.80}
NO_HEADROOM_CELLS = 1          # flip band <= this many cells => "no headroom" (contract s3)

# --- readings -------------------------------------------------------------

def t50_10(curve):
    """First 10-step checkpoint with ASR >= THR. Port of sf.t50_10; gate GS1-0 asserts
    it reproduces the published values on the real files."""
    for s, a in sorted(curve):
        if s % GRID == 0 and a >= THR:
            return s
    return None


def t50_sustained(curve, hold=HOLD):
    """First 10-step checkpoint with ASR >= THR such that every checkpoint within `hold`
    steps after it is also >= THR. Port of sf.t50_sustained."""
    pts = [(s, a) for s, a in sorted(curve) if s % GRID == 0]
    for i, (s, a) in enumerate(pts):
        if a >= THR and all(b >= THR for t, b in pts[i:] if t <= s + hold):
            return s
    return None


# --- binary budget --------------------------------------------------------

def installed(t50, budget):
    """The contract's dependent variable: did the behaviour reach the hit threshold
    within budget. None (never crossed on the grid) counts as not installed."""
    return t50 is not None and t50 <= budget


def installed_fraction(t50s, budget):
    return sum(installed(t, budget) for t in t50s), len(t50s)


def flip_band(t50s, budgets):
    """The flip band is the SET of grid budgets at which the arms do not all agree.
    Its size is reported in budgets (`cells`), NOT as hi-lo: counting budgets makes the
    band size equal the t50 span in cells, so one number serves both. Reporting hi-lo
    instead gives a number one smaller and was the source of a v2-plan inconsistency.
    Returns None when the arms agree at every budget on the axis."""
    dis = [b for b in budgets
           if 0 < sum(installed(t, b) for t in t50s) < len(t50s)]
    if not dis:
        return None
    lo, hi = min(dis), max(dis)
    return {"lo": lo, "hi": hi, "cells": len(dis), "endpoint_diff": hi - lo,
            "n_budgets_on_axis": len(list(budgets))}


def span(t50s):
    v = [t for t in t50s if t is not None]
    if len(v) < 2:
        return None
    return {"min": min(v), "max": max(v), "width": max(v) - min(v),
            "cells": (max(v) - min(v)) // GRID, "sd": sd(v)}


# --- statistics -----------------------------------------------------------

def mean(v):
    return sum(v) / len(v)


def sd(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def ranks(v):
    """Average ranks, ties shared."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def perm_p_two_sided(x, y, draws=10000, seed=0):
    """Permutation null for |spearman|. Returns (rho, p) or (None, None) if rho undefined."""
    import random
    rho = spearman(x, y)
    if rho is None:
        return None, None
    rng = random.Random(seed)
    yy = list(y)
    hit = 0
    for _ in range(draws):
        rng.shuffle(yy)
        r = spearman(x, yy)
        if r is not None and abs(r) >= abs(rho) - 1e-12:
            hit += 1
    return rho, (hit + 1) / (draws + 1)


# --- defender cost (2606.29554 section 3 Eq. 18) --------------------------

def seed_budget(sigma, delta):
    """n* per side >= 2 sigma^2 (z_{1-a/2}+z_{1-b})^2 / delta^2. Published criterion;
    only legitimate on the axis sigma was measured on."""
    return 2.0 * sigma ** 2 * Z_SUM ** 2 / delta ** 2


# --- loading --------------------------------------------------------------

def load_curves(path):
    """input_xo_behavioral.json -> {arm: [(step, asr), ...]}, {arm: [(step, clean_acc), ...]}"""
    d = json.load(open(path))
    asr, clean = {}, {}
    for arm, pts in d["arms"].items():
        asr[arm] = sorted((int(s), v["asr"]) for s, v in pts.items())
        clean[arm] = sorted((int(s), v["clean_acc"]) for s, v in pts.items())
    return asr, clean, d["source"]
