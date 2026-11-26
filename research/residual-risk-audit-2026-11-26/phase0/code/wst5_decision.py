"""WST5: decision-usability of the release rule U <= T (plan: phase0/PLAN.zh-CN.md, WST5 section, frozen 0047ae35).

Usage: python wst5_decision.py {rehearsal|main} out.json
"""
import json, sys
import numpy as np
from scipy import stats
from scipy.special import expit
from multiprocessing import Pool

T = 400
REPS = 300
N_CLEAN = 100_000
A_GRID = np.round(np.arange(-1.0, 9.01, 0.25), 2)


def cp_lower(x, n, a=0.05):
    return 0.0 if x == 0 else float(stats.beta.ppf(a, x, n - x + 1))


def binom_inversion(d, rL, a=0.05, cap=10 ** 7):
    if rL <= 0:
        return float("inf")
    lo, hi = d, d
    while stats.binom.cdf(d, hi, rL) >= a and hi < cap:
        hi = hi * 2 + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if stats.binom.cdf(d, mid, rL) >= a:
            lo = mid
        else:
            hi = mid
    return lo - d


def odds(c11, c10, c01, c00):
    return (c11 + 0.5) * (c00 + 0.5) / ((c10 + 0.5) * (c01 + 0.5))


def pi_upper(c, n11, n10, n01, rng, B=200):
    """c = canary 2x2 counts (c11, c10, c01, c00); multinomial bootstrap of the canary table."""
    K = sum(c)
    bs = rng.multinomial(K, np.array(c, float) / K, size=B)
    ors = odds(bs[:, 0], bs[:, 1], bs[:, 2], bs[:, 3])
    xi = float(np.quantile(ors, 0.95))
    U = xi * (n10 + 0.5) * (n01 + 0.5) / (n11 + 0.5)
    return U + 1.645 * np.sqrt(U) + 1


def union_recall(a, beta, n=400_000, seed=0):
    h = np.random.default_rng(seed).normal(0, 1, n)
    p = expit(a - beta * h)
    return float(np.mean(1 - (1 - p) ** 2))


def cell(args):
    beta, m, K, a, fp, seed = args
    rng = np.random.default_rng(seed)
    N = int(round(m * T))
    out = {k: [] for k in ("r", "bi", "pi", "mx")}
    for _ in range(REPS):
        h = rng.normal(0, 1, N)
        p = expit(a - beta * h)
        f1 = rng.random(N) < p
        f2 = rng.random(N) < p
        n11, n10, n01 = int(np.sum(f1 & f2)), int(np.sum(f1 & ~f2)), int(np.sum(~f1 & f2))
        r = N - n11 - n10 - n01
        if fp > 0:
            c1 = rng.random(N_CLEAN) < fp
            c2 = rng.random(N_CLEAN) < fp
            n11 += int(np.sum(c1 & c2)); n10 += int(np.sum(c1 & ~c2)); n01 += int(np.sum(~c1 & c2))
        hc = rng.normal(0, 1, K)
        pc = expit(a - beta * hc)
        g1 = rng.random(K) < pc
        g2 = rng.random(K) < pc
        c = (int(np.sum(g1 & g2)), int(np.sum(g1 & ~g2)), int(np.sum(~g1 & g2)), int(np.sum(~g1 & ~g2)))
        rL = cp_lower(K - c[3], K)
        ubi = binom_inversion(n11 + n10 + n01, rL)
        upi = pi_upper(c, n11, n10, n01, rng)
        out["r"].append(r); out["bi"].append(ubi); out["pi"].append(upi); out["mx"].append(max(ubi, upi))
    r = np.array(out["r"])
    res = {"beta": beta, "m": m, "K": K, "a": a, "fp": fp, "union_recall": union_recall(a, beta),
           "mean_residual_over_T": float(r.mean() / T)}
    for k in ("bi", "pi", "mx"):
        U = np.array(out[k])
        rel = U <= T
        res[f"P_release_{k}"] = float(rel.mean())
        res[f"unsafe_given_release_{k}"] = float(np.mean(r[rel] > T)) if rel.any() else None
        res[f"coverage_{k}"] = float(np.mean(r <= U))
    return res


def r_star(rows, method):
    """Smallest union recall with P(release) >= 0.8 (linear interpolation on the a-grid); unsafe rate at crossing."""
    rows = sorted(rows, key=lambda x: x["union_recall"])
    prev = None
    for x in rows:
        if x[f"P_release_{method}"] >= 0.8:
            if prev is None:
                rs = x["union_recall"]
            else:
                p0, p1 = prev[f"P_release_{method}"], x[f"P_release_{method}"]
                w = (0.8 - p0) / (p1 - p0) if p1 > p0 else 1.0
                rs = prev["union_recall"] + w * (x["union_recall"] - prev["union_recall"])
            return rs, x[f"unsafe_given_release_{method}"]
        prev = x
    return None, None


def verdict(rs):
    if rs is None or rs > 0.99:
        return "unusable"
    return "usable" if rs <= 0.95 else "marginal"


def best(rows, methods=("bi", "pi", "mx")):
    cands = []
    for mth in methods:
        rs, uns = r_star(rows, mth)
        if rs is not None and (uns is None or uns <= 0.05):
            cands.append((rs, mth, uns))
    if not cands:
        return {"r_star": None, "method": None, "unsafe": None, "verdict": "unusable"}
    rs, mth, uns = min(cands)
    return {"r_star": rs, "method": mth, "unsafe": uns, "verdict": verdict(rs)}


def grid(specs, seed0):
    jobs = []
    for i, (beta, m, K, fp) in enumerate(specs):
        for j, a in enumerate(A_GRID):
            jobs.append((beta, m, K, float(a), fp, seed0 + i * 1000 + j))
    with Pool(30) as pool:
        rows = pool.map(cell, jobs)
    out = {}
    for (beta, m, K, fp) in specs:
        out[f"b{beta}_m{m}_K{K}_fp{fp}"] = [x for x in rows if (x["beta"], x["m"], x["K"], x["fp"]) == (beta, m, K, fp)]
    return out


if __name__ == "__main__":
    mode, outp = sys.argv[1], sys.argv[2]
    if mode == "rehearsal":
        g = grid([(0.0, 2, 1000, 0.0), (2.0, 10, 200, 0.0)], 100)
        ra = best(g["b0.0_m2_K1000_fp0.0"])
        rb = best(g["b2.0_m10_K200_fp0.0"], methods=("pi",))
        ok = ra["verdict"] == "usable" and rb["verdict"] in ("unusable", "marginal")
        r = {"usable_demo": ra, "unusable_demo_pi_only": rb, "rehearsal_pass": ok, "grids": g}
        print(json.dumps({"usable_demo": ra, "unusable_demo_pi_only": rb, "rehearsal_pass": ok}, indent=1))
    else:
        specs = [(b, m, K, 0.0) for b in (0.0, 1.0, 2.0) for m in (1.5, 2, 5, 10) for K in (200, 1000)]
        specs.append((2.0, 5, 1000, 0.001))
        g = grid(specs, 5000)
        summary = {}
        for k, rows in g.items():
            per = {mth: dict(zip(("r_star", "unsafe"), r_star(rows, mth))) for mth in ("bi", "pi", "mx")}
            summary[k] = {"best": best(rows), "per_method": per}
        main = summary["b2.0_m5_K1000_fp0.0"]["best"]
        r = {"main_verdict": main, "summary": summary, "grids": g}
        print(json.dumps({"main_verdict": main}, indent=1))
        for k, v in summary.items():
            pm = v["per_method"]
            print(k, "best", v["best"]["verdict"], v["best"]["method"],
                  None if v["best"]["r_star"] is None else round(v["best"]["r_star"], 4),
                  "| bi", None if pm["bi"]["r_star"] is None else round(pm["bi"]["r_star"], 4),
                  "pi", None if pm["pi"]["r_star"] is None else round(pm["pi"]["r_star"], 4),
                  "mx", None if pm["mx"]["r_star"] is None else round(pm["mx"]["r_star"], 4))
    json.dump(r, open(outp, "w"), indent=1, default=float)
