"""Qwen 1% crossover: does order or init decide WHETHER the backdoor installs? (CONTRACT_QX.md)"""
import numpy as np

N = 20000
CROSS = [(1004, 1001), (1005, 1001), (1001, 1004), (1001, 1005)]   # (init, order)
BLIND = [(1001, 1009), (1001, 1010)]                              # LLaVA's two earliest orders
SEED_KEY = {1001: "s1", 1004: "s4", 1005: "s5"}
EVAL = list(range(200, 1241, 40))                                 # + the final adapter (1250)
INSTALL, FAIL = 0.9, 0.1


def arm(init, order):
    return f"QX-I{init}-O{order}"


def runs():
    return [CROSS[0], CROSS[2], CROSS[1], CROSS[3]] + BLIND      # interleave the two directions


def src_map(r_init, r_order):
    src = np.empty(N, dtype=np.int64)
    src[np.asarray(r_init)] = np.asarray(r_order)
    return src


def outcome(curve):
    """curve: [(step, asr)] including step 1250 (final adapter)."""
    if not curve:
        return None
    d = dict(curve)
    if 1250 not in d:
        return None
    if d[1250] >= INSTALL:
        return "install"
    if max(d.values()) <= FAIL:
        return "fail"
    return "partial"


def cross_verdict(o):
    """o: {(init, order): outcome}. Order hypothesis: install iff order == 1001."""
    v = [o.get(k) for k in CROSS]
    if any(x is None for x in v):
        return "cannot measure"
    a, b, c, d = v                                   # I1004-O1001, I1005-O1001, I1001-O1004, I1001-O1005
    if a == b == "install" and c == d == "fail":
        return "order decides whether it installs"
    if a == b == "fail" and c == d == "install":
        return "init decides whether it installs"
    if all(x == "fail" for x in v):
        return "neither alone: 1001 needs its own init and order together"
    if all(x == "install" for x in v):
        return "both swaps install (1004/1005 failures need their own init and order together)"
    return "mixed"


def blind_verdict(o):
    """Amended before any QX result (LOG J41): a cross-model transfer clue only. No probability is attached:
    the historical 1/5 install rate varied init and order together, while these runs fix init 1001."""
    v = [o.get(k) for k in BLIND]
    if any(x is None for x in v):
        return {"tier": "cannot measure"}
    n = sum(x == "install" for x in v)
    tier = ("clue: both LLaVA-early orders install on Qwen" if n == 2
            else "clue against: both LLaVA-early orders fail on Qwen" if all(x == "fail" for x in v)
            else "unclear")
    return {"n_install": n, "tier": tier}


def step1_gate(loss, tol=1e-4, ctrl_min=1e-3):
    a = abs(loss["S1004-ORIG"] - loss["QX-I1001-O1004"])
    b = abs(loss["S1001-ORIG"] - loss["QX-I1004-O1001"])
    c = abs(loss["S1004-ORIG"] - loss["S1001-ORIG"])
    return {"pair_1004": a, "pair_1001": b, "control": c, "pass": a <= tol and b <= tol and c > ctrl_min}
