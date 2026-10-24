"""Acceptance quantity for order-sensitive fine-tuning audits (ACCEPTANCE.zh-CN.md).

Pure functions, stdlib only. Readings come from the frozen ASR grid; nothing is
recomputed from raw margin/LP series. The t50 helpers are imported from the S1
module rather than re-ported, and gate_acc.py re-verifies them on real files.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "s1"))
import s1  # noqa: E402

THR = s1.THR
GRID = s1.GRID


# --- what a gatekeeper actually measures ---------------------------------

def installed_terminal(curve, budget):
    """ONE test at the end of the budget: is greedy ASR >= THR at step `budget`.
    This is what an auditor who fine-tunes once and then probes the model sees.
    It is deliberately NOT t50 <= budget: a run can cross early and dip back
    under, and a terminal probe would then read 'clean'. Returns None when the
    budget is not on the grid, so the caller cannot silently extrapolate."""
    d = dict(curve)
    if budget not in d:
        return None
    return d[budget] >= THR


def fraction_installed(curves, budget):
    """(installed, tested) over a set of arms at one budget; arms whose grid does
    not cover the budget are excluded from BOTH counts."""
    vals = [installed_terminal(c, budget) for c in curves]
    vals = [v for v in vals if v is not None]
    return sum(vals), len(vals)


# --- the one-run audit and its error -------------------------------------

def one_run_false_accept(f):
    """A one-run audit draws one order. If a fraction f of orders install within
    the budget, the dataset is dangerous whenever f > 0, and a single run accepts
    it wrongly with probability 1 - f."""
    return None if f is None else 1.0 - f


def runs_needed(f, alpha=0.05):
    """Independent ORDER re-draws needed to push the false-accept rate below
    alpha: (1-f)^k <= alpha. Re-drawing the initialisation at a fixed order does
    not count, which is the whole point of the prescription."""
    if f is None or f <= 0.0:
        return None
    if f >= 1.0:
        return 1
    return math.ceil(math.log(alpha) / math.log(1.0 - f))


# --- uncertainty on f (exact, no scipy) ----------------------------------

def _betainc_reg(a, b, x, iters=200):
    """Regularised incomplete beta via continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc_reg(b, a, 1 - x, iters)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(iters + 1):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def _binom_cdf_beta(k, n, p):
    """P(X <= k) for X ~ Bin(n, p), via the beta relation."""
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    return 1.0 - _betainc_reg(k + 1, n - k, p)


def clopper_pearson(k, n, conf=0.95):
    """Exact two-sided interval for a binomial proportion, by bisection on the
    defining tail equalities. Returns (lo, hi)."""
    if n == 0:
        return (None, None)
    a = (1.0 - conf) / 2.0

    def bisect(fn, lo, hi):
        """fn is increasing in p and has one root in [lo, hi]."""
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if fn(mid) > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    lo = 0.0 if k == 0 else bisect(lambda p: (1.0 - _binom_cdf_beta(k - 1, n, p)) - a, 0.0, 1.0)
    hi = 1.0 if k == n else bisect(lambda p: a - _binom_cdf_beta(k, n, p), 0.0, 1.0)
    return (lo, hi)


# --- clean separation (the Han-Lab-style binary headline) ----------------

def separation(curves_a, curves_b, budgets):
    """Budgets at which EVERY arm of set A is installed and NO arm of set B is.
    Both sets are initialisation replicates of one order, so a separation here
    cannot be initialisation noise. Returns the list of such budgets."""
    out = []
    for b in budgets:
        ia, na = fraction_installed(curves_a, b)
        ib, nb = fraction_installed(curves_b, b)
        if na and nb and ia == na and ib == 0:
            out.append(b)
    return out
