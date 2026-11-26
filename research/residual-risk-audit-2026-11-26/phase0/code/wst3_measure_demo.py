"""WST3: synthetic demo of the measurement layer (plan: phase0/PLAN.zh-CN.md, frozen 01486c25).

Two filters, latent hardness h, detection p_j(h) = expit(a_j - beta*h), filters independent given h.
Methods: Chapman (Lincoln-Petersen) CI; canary-calibrated partial identification (simplified);
binomial inversion of spike-in recall; M1 falsification audit. 1000 replicates per scenario.
Usage: python wst3_measure_demo.py out.json
"""
import json, sys
import numpy as np
from scipy import stats
from scipy.special import expit

REPS = 1000
P_TRUE = 500
N_CLEAN = 100_000
K_CANARY = 200
A1, A2 = 0.4, 0.0  # filter intercepts (recall ~0.6 / ~0.5 at beta=0)


def detect(h, beta, hidden, rng):
    p1 = expit(A1 - beta * h)
    p2 = expit(A2 - beta * h)
    p1 = np.where(hidden, 0.0, p1)
    p2 = np.where(hidden, 0.0, p2)
    return rng.random(h.size) < p1, rng.random(h.size) < p2


def table(f1, f2):
    return int(np.sum(f1 & f2)), int(np.sum(f1 & ~f2)), int(np.sum(~f1 & f2)), int(np.sum(~f1 & ~f2))


def chapman(n11, n10, n01):
    n1, n2 = n11 + n10, n11 + n01
    N = (n1 + 1) * (n2 + 1) / (n11 + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - n11) * (n2 - n11) / ((n11 + 1) ** 2 * (n11 + 2))
    obs = n11 + n10 + n01
    se = np.sqrt(var)
    return N - obs, N - 1.96 * se - obs, N + 1.96 * se - obs


def odds(c11, c10, c01, c00):
    return (c11 + 0.5) * (c00 + 0.5) / ((c10 + 0.5) * (c01 + 0.5))


def pi_upper(canary_f1, canary_f2, n11, n10, n01, rng, B=200):
    K = canary_f1.size
    ors = []
    for _ in range(B):
        idx = rng.integers(0, K, K)
        ors.append(odds(*table(canary_f1[idx], canary_f2[idx])))
    xi = float(np.quantile(ors, 0.95))
    U = xi * (n10 + 0.5) * (n01 + 0.5) / (n11 + 0.5)
    return xi, U + 1.645 * np.sqrt(U) + 1  # Poisson-style inflation (simplified; not Sun et al.)


def cp_lower(x, n, a=0.05):
    return 0.0 if x == 0 else float(stats.beta.ppf(a, x, n - x + 1))


def cp_upper(x, n, a=0.05):
    return 1.0 if x == n else float(stats.beta.ppf(1 - a, x + 1, n - x))


def binom_inversion(d, rL, a=0.05, cap=10 ** 6):
    if rL <= 0:
        return float("inf")
    lo, hi = d, d
    while stats.binom.cdf(d, hi, rL) >= a and hi < cap:
        hi = hi * 2 + 1
    while hi - lo > 1:  # largest P with cdf >= a
        mid = (lo + hi) // 2
        if stats.binom.cdf(d, mid, rL) >= a:
            lo = mid
        else:
            hi = mid
    return lo - d


def scenario(name, beta, hidden_frac, canary_shift, fp, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(REPS):
        h = rng.normal(0, 1, P_TRUE)
        hidden = rng.random(P_TRUE) < hidden_frac
        f1, f2 = detect(h, beta, hidden, rng)
        n11, n10, n01, n00 = table(f1, f2)
        # unverified flags: clean false positives add to the flag tables
        if fp > 0:
            c1 = rng.random(N_CLEAN) < fp
            c2 = rng.random(N_CLEAN) < fp
            m11, m10, m01, _ = table(c1, c2)
            o11, o10, o01 = n11 + m11, n10 + m10, n01 + m01
        else:
            o11, o10, o01 = n11, n10, n01
        est, lo, hi = chapman(o11, o10, o01)
        hc = rng.normal(canary_shift, 1, K_CANARY)
        cf1, cf2 = detect(hc, beta, np.zeros(K_CANARY, bool), rng)
        xi, U_pi = pi_upper(cf1, cf2, o11, o10, o01, rng)
        # binomial inversion: union-filter recall on canaries
        cu = int(np.sum(cf1 | cf2))
        rL = cp_lower(cu, K_CANARY)
        d = o11 + o10 + o01
        U_bi = binom_inversion(d, rL)
        rows.append({"n00": n00, "chap": est, "chap_lo": lo, "chap_hi": hi, "xi": xi, "U_pi": U_pi, "U_bi": U_bi,
                     "naive": d / max(rL, 1e-9) - d, "or_real": odds(n11, n10, n01, n00)})
    n00 = np.array([r["n00"] for r in rows])
    g = lambda k: np.array([r[k] for r in rows])
    return {"name": name, "beta": beta, "hidden_frac": hidden_frac, "canary_shift": canary_shift, "fp": fp,
            "mean_true_n00": float(n00.mean()), "mean_or_real": float(np.median(g("or_real"))),
            "median_xi": float(np.median(g("xi"))),
            "chapman_mean": float(g("chap").mean()),
            "chapman_coverage": float(np.mean((g("chap_lo") <= n00) & (n00 <= g("chap_hi")))),
            "pi_coverage": float(np.mean(n00 <= g("U_pi"))), "pi_median_upper": float(np.median(g("U_pi"))),
            "bi_coverage": float(np.mean(n00 <= g("U_bi"))), "bi_median_upper": float(np.median(g("U_bi"))),
            "naive_coverage": float(np.mean(n00 <= g("naive")))}


def m1_audit(true_recall, K=100, r0=0.90, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(reps):
        x = int(rng.binomial(K, true_recall))
        if cp_upper(x, K) < r0:
            fails += 1
    return fails / reps


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "wst3.json"
    S = {
        "S-a": scenario("S-a", 0.0, 0.0, 0.0, 0.0, 1),
        "S-b": scenario("S-b", 2.0, 0.0, 0.0, 0.0, 2),
        "S-c": scenario("S-c", 2.0, 0.10, 0.0, 0.0, 3),
        "S-d": scenario("S-d", 2.0, 0.0, -1.0, 0.0, 4),
        "S-b_fp": scenario("S-b_fp", 2.0, 0.0, 0.0, 0.001, 5),
    }
    d0 = {rl: binom_inversion(0, rl) for rl in (0.5, 0.8, 0.95)}
    m1 = {"recall_0.60": m1_audit(0.60, seed=11), "recall_0.95": m1_audit(0.95, seed=12)}
    C1 = S["S-b"]["chapman_coverage"] < 0.80 and S["S-a"]["chapman_coverage"] >= 0.90
    C2 = S["S-b"]["pi_coverage"] >= 0.93 and S["S-d"]["pi_coverage"] < 0.90
    C4 = all(v > 0 for v in d0.values()) and S["S-b"]["bi_coverage"] >= 0.93
    C5 = m1["recall_0.60"] >= 0.95 and m1["recall_0.95"] <= 0.05
    r = {"scenarios": S, "binom_inversion_d0": d0, "m1_fail_rate": m1,
         "criteria": {"C1": C1, "C2": C2, "C4": C4, "C5": C5}}
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(json.dumps(r, indent=1, default=float))
