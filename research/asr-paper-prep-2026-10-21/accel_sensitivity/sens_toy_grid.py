"""Sensitivity of the in-window / pre-window acceleration ratio on the three Q1 toys to:
quantity (margin T3 vs gap), aggregate (median vs mean), window rule (0.05/0.95 vs 0.10/0.90),
grid (every step vs every 20 steps, all 20 offsets), pre-window (equal-length vs anchored at step 20).
Descriptive only; reuses the r2 / atlas two-point slope definitions verbatim."""
import json
import numpy as np

ROOT = "/home/ubuntu/.claude/jobs/f99cf25d/tmp/accel/toy/"


def load(s):
    rows = [json.loads(l) for l in open(ROOT + f"margins_{s}.jsonl")]
    st = np.array([r["update"] for r in rows])
    M = np.array([r["trigger"]["margin"] for r in rows])
    G = np.array([r["trigger"]["gap"] for r in rows])
    F = np.array([r["trigger"]["hit"] for r in rows])
    return st, M, G, F


def window(steps, asr, lo, hi):
    b = next((s for s, v in zip(steps, asr) if v >= hi), None)
    if b is None:
        return None
    pre = [s for s, v in zip(steps, asr) if s < b and v <= lo]
    return (pre[-1], b) if pre else None


def at(steps, s):
    c = [x for x in steps if x <= s]
    return c[-1] if c else steps[0]


def run(st, X, F, agg, lo, hi, anchor=None):
    steps = list(st)
    asr = F.mean(1)
    w = window(steps, asr, lo, hi)
    if w is None:
        return None
    a, b = w
    p = at(steps, a - (b - a)) if anchor is None else at(steps, anchor)
    if p >= a:
        return None
    c = np.median(X, 1) if agg == "median" else X.mean(1)
    ia, ib, ip = steps.index(a), steps.index(b), steps.index(p)
    vin = (c[ib] - c[ia]) / (b - a)
    vpre = (c[ia] - c[ip]) / max(1, a - p)
    R = vin / vpre if vpre > 0 else float("inf")
    sin = (X[ib] - X[ia]) / (b - a)
    spre = (X[ia] - X[ip]) / max(1, a - p)
    d3 = float(np.where(spre > 0, sin > 1.5 * spre, sin > 0).mean())
    agree = float(((X[ip:ib + 1] > 0) == F[ip:ib + 1]).mean())
    return dict(p=p, a=a, b=b, R=R, D3=d3, agree=agree)


def sub(st, X, F, off):
    k = (st % 20 == off) | (st == st[-1])
    return st[k], X[k], F[k]


def f(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return "inf" if x == float("inf") else f"{x:.2f}"
    return str(x)


rows = []
for s in ("s1105", "s1106", "s1107"):
    st, M, G, F = load(s)
    for qn, Q in (("margin", M), ("gap", G)):
        for agg in ("median", "mean"):
            for lo, hi in ((0.05, 0.95), (0.10, 0.90)):
                r = run(st, Q, F, agg, lo, hi)
                r20 = [run(*sub(st, Q, F, o), agg, lo, hi) for o in range(20)]
                r20 = [x for x in r20 if x]
                Rs = [x["R"] for x in r20]
                fin = [x for x in Rs if x != float("inf")]
                ra = run(st, Q, F, agg, lo, hi, anchor=20)
                rows.append((s, qn, agg, f"{lo}/{hi}", r, (np.median(fin) if fin else None, min(fin) if fin else None,
                                                             max(fin) if fin else None, len(Rs), len(Rs) - len(fin)), ra))

print("toy    qty    agg    win       | 1-step grid: p→[a,b]  R  D3  agree          | 20-step grid R: med [min,max] (n, n_inf) | pre-window anchored at step 20")
for s, qn, agg, w, r, g, ra in rows:
    r1 = f"{r['p']}→[{r['a']},{r['b']}] R={f(r['R'])} D3={f(r['D3'])} ag={f(r['agree'])}" if r else "no window"
    g1 = f"{f(g[0])} [{f(g[1])},{f(g[2])}] ({g[3]},{g[4]})" if g[3] else "—"
    a1 = f"p={ra['p']} R={f(ra['R'])} D3={f(ra['D3'])}" if ra else "—"
    print(f"{s:6} {qn:6} {agg:6} {w:9} | {r1:46} | {g1:30} | {a1}")

# gap > 0 vs hit identity check (is the atlas gate vacuous for gap?)
for s in ("s1105", "s1106", "s1107"):
    st, M, G, F = load(s)
    print(s, "agree(gap>0, hit) =", float(((G > 0) == F).mean()), " agree(margin>0, hit) =", float(((M > 0) == F).mean()),
          " median gap - margin at a..b:", f(float(np.median(G - M))))
