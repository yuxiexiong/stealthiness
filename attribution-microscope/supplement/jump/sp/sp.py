"""Splice experiment + sequence-feature exploration (CONTRACT_SP.md). numpy only."""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jump as J  # noqa: E402

N, BATCH, H = 20000, 16, 300
INIT = 1001                                       # every splice run uses init (seed) 1001
A_, B_ = 1007, 1001                               # the two orders farthest apart at init 1001
Y = {1007: 470, 1001: 360}                        # t50 (10-step grid) at init 1001: XO-I1001-O1007, s1
CUTS = (200, 100, 300)                            # queue order: middle cut first
# t50 per order for the exploration (pre-specified): init-1001 value where measured, else own seed
T50 = {1001: 360, 1007: 470, 1008: 410, 1011: 410, 1009: 330, 1004: 370, 1005: 390, 1006: 390, 1010: 350}
FEATURES = ("K300", "D300", "L30", "L100", "Burst20", "Warm38", "OpenShare300", "MeanArrival300")
SHARE_HI, SHARE_LO = 0.75, 0.25


def arm(prefix, suffix, c):
    return f"SP-A{prefix}-B{suffix}-C{c}"


def runs():
    out = []
    for c in CUTS:
        out += [(A_, B_, c), (B_, A_, c)]
    return out


def splice(seq_prefix, seq_suffix, c):
    """First 16c consumed rows from seq_prefix, then every remaining row in seq_suffix's order."""
    head = np.asarray(seq_prefix)[:BATCH * c]
    used = np.zeros(N, bool)
    used[head] = True
    tail = np.asarray([r for r in seq_suffix if not used[r]])
    out = np.concatenate([head, tail])
    assert len(out) == N and len(np.unique(out)) == N
    return out


# ---------------------------------------------------------------- features
def features(seq, rows, open_rows):
    """seq: consumed row sequence; rows: poison rows; open_rows: poison rows whose clean answer was open."""
    st = J.row_steps(seq, rows)
    st = st[st <= H]
    op = J.row_steps(seq, sorted(open_rows))
    op = op[op <= H]
    w = np.bincount((st - 1) // 20, minlength=H // 20)
    return {"K300": float(st.size),
            "D300": float(sum(J.lam(k - 1) for k in st)),
            "L30": float(np.exp(-(H - st) / 30).sum()),
            "L100": float(np.exp(-(H - st) / 100).sum()),
            "Burst20": float(w.max()) if st.size else 0.0,
            "Warm38": float((st <= 38).sum()),
            "OpenShare300": float(op.size / st.size) if st.size else 0.0,
            "MeanArrival300": float(st.mean()) if st.size else 0.0}


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


_PERMS = {}


def perm_p2(x, y):
    """Two-sided exact permutation p of Spearman rho (all n! assignments, vectorised)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = spearman(x, y)
    n = len(x)
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return r, 1.0
    zx = (rx - rx.mean()) / rx.std()
    zy = (ry - ry.mean()) / ry.std()
    null = zx[_PERMS[n]] @ zy / n
    return r, float((np.abs(null) >= abs(r) - 1e-9).mean())


def holm(ps):
    """Holm-adjusted p-values (same order as input)."""
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def explore(feat_by_order):
    """feat_by_order: {order: {feature: value}}. Spearman with T50 per feature, exact p, Holm."""
    orders = sorted(T50)
    y = [T50[o] for o in orders]
    out = {}
    for f in FEATURES:
        x = [feat_by_order[o][f] for o in orders]
        r, p = perm_p2(x, y)
        out[f] = {"rho": r, "p": p}
    adj = holm([out[f]["p"] for f in FEATURES])
    for f, a in zip(FEATURES, adj):
        out[f]["p_holm"] = float(a)
        out[f]["tier"] = ("candidate" if a <= 0.05 else "clue" if out[f]["p"] <= 0.05 else "none")
    primary = min(FEATURES, key=lambda f: (out[f]["p"], FEATURES.index(f)))
    return out, primary


def fit_predict(feat_by_order, feat_new, f):
    """Least-squares t50 = a + b * f on the nine orders; predictions for new sequences."""
    orders = sorted(T50)
    x = np.array([feat_by_order[o][f] for o in orders])
    y = np.array([T50[o] for o in orders], float)
    if x.std() == 0:
        return {k: float(y.mean()) for k in feat_new}
    b, a = np.polyfit(x, y, 1)
    return {k: float(a + b * v[f]) for k, v in feat_new.items()}


def blind_tier(pred, obs):
    """Primary feature on the splice runs: MAE vs the constant mean of the nine orders."""
    ks = [k for k in pred if obs.get(k) is not None]
    if len(ks) < 4:
        return {"n": len(ks), "tier": "cannot measure"}
    const = float(np.mean(list(T50.values())))
    mf = float(np.mean([abs(pred[k] - obs[k]) for k in ks]))
    mc = float(np.mean([abs(const - obs[k]) for k in ks]))
    ratio = mf / mc if mc > 0 else float("inf")
    tier = ("passes blind test" if ratio <= 0.5 else "fails blind test" if ratio >= 1 else "unclear")
    return {"n": len(ks), "mae_feature": mf, "mae_const": mc, "ratio": ratio, "tier": tier}


# ---------------------------------------------------------------- splice verdict
def share(t, prefix, suffix):
    return (t - Y[suffix]) / (Y[prefix] - Y[suffix])


def label(s):
    if s is None:
        return None
    return "prefix" if s >= SHARE_HI else "suffix" if s <= SHARE_LO else "mixed"


def localize(obs):
    """obs: {(prefix, suffix, c): t50 or None}. Where does the deciding segment lie?"""
    lab, sh = {}, {}
    for (p, s, c), t in obs.items():
        sh[(p, s, c)] = None if t is None else share(t, p, s)
        lab[(p, s, c)] = label(sh[(p, s, c)])
    if any(v is None for v in lab.values()) or len(lab) < 6:
        return {"tier": "cannot measure (missing runs)", "labels": lab, "shares": sh}
    dirs = [(A_, B_), (B_, A_)]
    cs = sorted(CUTS)
    L = {d: [lab[(d[0], d[1], c)] for c in cs] for d in dirs}
    S = {d: [sh[(d[0], d[1], c)] for c in cs] for d in dirs}
    tier = "inconsistent"
    if all(L[d][0] == "prefix" for d in dirs):
        tier = f"deciding segment within steps 1-{cs[0]}"
    elif all(L[d][-1] == "suffix" for d in dirs):
        tier = f"deciding segment after step {cs[-1]}"
    else:
        for i in range(len(cs) - 1):
            if all(L[d][i] == "suffix" and L[d][i + 1] == "prefix" for d in dirs):
                tier = f"deciding segment within steps {cs[i] + 1}-{cs[i + 1]}"
        if tier == "inconsistent" and all(all(b >= a - SHARE_LO for a, b in zip(S[d], S[d][1:])) for d in dirs) \
                and any("mixed" in L[d] for d in dirs):
            tier = "graded (no single deciding segment)"
    return {"tier": tier, "labels": {f"{k[0]}>{k[1]}@{k[2]}": v for k, v in lab.items()},
            "shares": {f"{k[0]}>{k[1]}@{k[2]}": v for k, v in sh.items()}}


def step1_gate(loss, tol=1e-4):
    """Step-1 loss of one splice run per direction equals the known XO gate loss of its prefix order."""
    a = abs(loss["SP-A1007"] - 4.0838)
    b = abs(loss["SP-A1001"] - 4.7161)
    return {"d1007": a, "d1001": b, "pass": a <= tol and b <= tol}
