"""WST2: BMD/BMDL refit of public LLM backdoor dose-response data (plan: phase0/PLAN.zh-CN.md, frozen 01486c25).

Usage: python wst2_bmdl.py {rehearsal|real|budget} [out.json]
Doses are poisoned-example counts; internally scaled by 1/1000.
"""
import json, sys, math
import numpy as np
from scipy import optimize, stats
from scipy.special import expit, logit

BMR = 0.10
CRIT = stats.chi2.ppf(0.90, 1)  # one-sided 95% profile bound: 2.706
SC = 1000.0

# Zheng & Zen table 1 (20,000 examples; dose = poisoned count)
ZHENG = [(20, 0, 3), (100, 0, 3), (200, 0, 3), (400, 0, 3), (600, 0, 3),
         (700, 13, 20), (800, 6, 20), (810, 9, 10), (820, 18, 20), (830, 2, 10), (840, 8, 20),
         (850, 10, 10), (860, 10, 10), (880, 10, 10), (900, 10, 10), (1000, 10, 10)]

EPS = 1e-9


def ll(p, y, n):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(y * np.log(p) + (n - y) * np.log(1 - p)))


# Each model: k params; prob(free, x, d) gives P(d) with BMD constrained to x (scaled units).
# free params are unconstrained reals mapped inside.
def m_logistic(f, x, d):
    a = f[0]
    p0 = expit(a)
    t = logit(p0 + BMR * (1 - p0))
    b = (t - a) / x
    return expit(a + b * d)


def m_probit(f, x, d):
    a = f[0]
    p0 = stats.norm.cdf(a)
    t = stats.norm.ppf(p0 + BMR * (1 - p0))
    b = (t - a) / x
    return stats.norm.cdf(a + b * d)


def m_loglogistic(f, x, d):
    g, b = expit(f[0]), 18.0 * expit(f[1])  # slope in (0, 18] (BMDS bound); DECISIONS P1
    a = logit(BMR) - b * math.log(x)
    return g + (1 - g) * expit(a + b * np.log(d))


def m_logprobit(f, x, d):
    g, b = expit(f[0]), 18.0 * expit(f[1])  # slope in (0, 18] (BMDS bound); DECISIONS P1
    a = stats.norm.ppf(BMR) - b * math.log(x)
    return g + (1 - g) * stats.norm.cdf(a + b * np.log(d))


def m_weibull(f, x, d):
    g, al = expit(f[0]), 1.0 + 17.0 * expit(f[1])  # alpha in [1, 18] (BMDS bound); DECISIONS P1
    beta = -math.log(1 - BMR) / x ** al
    return g + (1 - g) * (1 - np.exp(-beta * d ** al))


MODELS = {
    "logistic": (m_logistic, 1, [[-8.0], [-4.0], [-12.0]]),
    "probit": (m_probit, 1, [[-4.0], [-2.5], [-6.0]]),
    "loglogistic": (m_loglogistic, 2, [[-8, 1.0], [-6, 2.5], [-10, 0.0]]),
    "logprobit": (m_logprobit, 2, [[-8, 1.0], [-6, 2.0], [-10, 0.0]]),
    "weibull": (m_weibull, 2, [[-8, 1.0], [-6, 2.5], [-10, 0.0]]),
}


def prof(model, x, d, y, n):
    fn, kf, starts = MODELS[model]
    best = -np.inf
    bestf = None
    for s in starts:
        r = optimize.minimize(lambda f: -ll(fn(f, x, d), y, n), np.array(s, float), method="Nelder-Mead",
                              options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        if -r.fun > best:
            best, bestf = -r.fun, r.x
    return best, bestf


def fit(model, d, y, n):
    """MLE over (free params, log x); then profile-likelihood BMDL."""
    xs = np.exp(np.linspace(np.log(d.min() / 20), np.log(d.max() * 2), 40))
    pl = np.array([prof(model, x, d, y, n)[0] for x in xs])
    i = int(np.argmax(pl))
    # refine x around the grid max
    lo, hi = np.log(xs[max(i - 1, 0)]), np.log(xs[min(i + 1, len(xs) - 1)])
    r = optimize.minimize_scalar(lambda lx: -prof(model, math.exp(lx), d, y, n)[0], bounds=(lo, hi), method="bounded",
                                 options={"xatol": 1e-6})
    xhat = math.exp(r.x)
    lmax, fhat = prof(model, xhat, d, y, n)
    if pl[i] > lmax:
        xhat, lmax = xs[i], pl[i]
        fhat = prof(model, xhat, d, y, n)[1]
    g = lambda lx: 2 * (lmax - prof(model, math.exp(lx), d, y, n)[0]) - CRIT
    lo_x = math.log(xs[0])
    if g(lo_x) <= 0:
        bmdl = xs[0]
        bmdl_censored = True
    else:
        bmdl = math.exp(optimize.brentq(g, lo_x, math.log(xhat), xtol=1e-6))
        bmdl_censored = False
    fn, kf, _ = MODELS[model]
    p = fn(fhat, xhat, d)
    k = kf + 1
    pc = np.clip(p, EPS, 1 - EPS)
    chi = float(np.sum((y - n * pc) ** 2 / (n * pc * (1 - pc))))
    df = len(d) - k
    gof_p = float(stats.chi2.sf(chi, df)) if df > 0 else float("nan")
    return {"model": model, "k": k, "loglik": lmax, "aic": 2 * k - 2 * lmax, "bmd": xhat * SC, "bmdl": bmdl * SC,
            "bmdl_censored_at_grid_floor": bmdl_censored, "ratio": xhat / bmdl, "pearson_chi2": chi, "df": df,
            "gof_p": gof_p, "_free": [float(v) for v in fhat], "_xhat": xhat}


def select(fits):
    ok = [f for f in fits if f["gof_p"] >= 0.1]
    if not ok:
        return None, "no_adequate_model"
    bm = [f["bmdl"] for f in ok]
    if max(bm) / min(bm) <= 3:
        return min(ok, key=lambda f: f["aic"]), "min_aic"
    return min(ok, key=lambda f: f["bmdl"]), "min_bmdl"


def run_all(dose, y, n):
    d = np.asarray(dose, float) / SC
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    fits = [fit(m, d, y, n) for m in MODELS]
    sel, rule = select(fits)
    return fits, sel, rule


def verdicts(sel):
    j1 = "rewritten" if (sel is None or sel["bmdl"] < 600) else "not_rewritten"
    j2 = "n/a" if sel is None else ("usable" if sel["ratio"] <= 2 else "insufficient")
    return j1, j2


def pava(y, n):
    p = list(y / n)
    w = list(n.astype(float))
    blocks = [[p[i], w[i], 1] for i in range(len(p))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, c in blocks:
        out += [v] * c
    return np.array(out)


def dev_sat(y, n, p):
    p = np.clip(p, EPS, 1 - EPS)
    t1 = np.where(y > 0, y * np.log(np.maximum(y, EPS) / (n * p)), 0.0)
    t2 = np.where(n - y > 0, (n - y) * np.log(np.maximum(n - y, EPS) / (n * (1 - p))), 0.0)
    return float(2 * np.sum(t1 + t2))


def monotone_test(y, n, B=2000, seed=0):
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    piso = pava(y, n)
    dobs = dev_sat(y, n, piso)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        ys = rng.binomial(n.astype(int), np.clip(piso, 0, 1)).astype(float)
        if dev_sat(ys, n, pava(ys, n)) >= dobs:
            cnt += 1
    return {"deviance_iso_vs_sat": dobs, "p_boot": (cnt + 1) / (B + 1), "B": B, "p_iso": piso.tolist()}


def true_logistic(ed50, slope_per_100):
    b = slope_per_100 / 100.0
    return lambda d: expit(b * (np.asarray(d, float) - ed50))


def design(curve, nseed):
    # 6 doses equally spaced between ED10 and ED90 of the true curve
    grid = np.linspace(1, 3000, 30000)
    p = curve(grid)
    lo, hi = grid[np.argmax(p >= 0.1)], grid[np.argmax(p >= 0.9)]
    dose = np.linspace(lo, hi, 6)
    return dose, np.full(6, nseed)


def true_bmd(curve):
    grid = np.linspace(1, 3000, 300000)
    p = curve(grid)
    p0 = float(curve(0.0))
    er = (p - p0) / (1 - p0)
    return float(grid[np.argmax(er >= BMR)])


def _one(args):
    curve_spec, dose, n, models, s = args
    curve = true_logistic(*curve_spec)
    rng = np.random.default_rng(s)
    y = rng.binomial(n.astype(int), curve(dose)).astype(float)
    fs = [fit(m, dose / SC, y, n.astype(float)) for m in models]
    sel, rule = select(fs) if len(models) > 1 else (fs[0], "single")
    return {"ratio": sel["ratio"] if sel else float("nan"), "bmdl": sel["bmdl"] if sel else float("nan"),
            "sel_none": sel is None, "verdicts": verdicts(sel)}


def simulate(curve_spec, nseed, reps, seed, models=("logistic",)):
    from multiprocessing import Pool
    curve = true_logistic(*curve_spec)
    dose, n = design(curve, nseed)
    tb = true_bmd(curve)
    with Pool(30) as pool:
        res = pool.map(_one, [(curve_spec, dose, n, models, seed * 100000 + i) for i in range(reps)])
    r = np.array([x["ratio"] for x in res])
    b = np.array([x["bmdl"] for x in res])
    return {"nseed": nseed, "reps": reps, "doses": dose.tolist(), "true_bmd": tb,
            "P_ratio_le_2": float(np.nanmean(r <= 2)), "bmdl_coverage": float(np.nanmean(b <= tb)),
            "frac_no_adequate": float(np.mean([x["sel_none"] for x in res])),
            "J1_counts": {k: sum(1 for x in res if x["verdicts"][0] == k) for k in ("rewritten", "not_rewritten")},
            "J2_counts": {k: sum(1 for x in res if x["verdicts"][1] == k) for k in ("usable", "insufficient", "n/a")}}


def j2v2(sel, dose, n, B=200, seed=7, pool=None):
    """Parametric bootstrap at the actual design; refit only the selected model."""
    if sel is None:
        return {"J2v2": "n/a"}
    fn = MODELS[sel["model"]][0]
    d = np.asarray(dose, float) / SC
    n = np.asarray(n, float)
    p = np.clip(fn(np.array(sel["_free"]), sel["_xhat"], d), 0, 1)
    rng = np.random.default_rng(seed)
    ys = [rng.binomial(n.astype(int), p).astype(float) for _ in range(B)]
    args = [(sel["model"], d, y, n) for y in ys]
    res = pool.map(_refit, args) if pool is not None else [_refit(a) for a in args]
    bm = np.array([r[0] for r in res]); ra = np.array([r[1] for r in res])
    cov = float(np.mean(bm <= sel["bmd"]))
    med = float(np.median(ra))
    return {"J2v2": "usable" if (cov >= 0.90 and med <= 2) else "insufficient", "boot_coverage": cov,
            "boot_median_ratio": med, "B": B}


def _refit(a):
    m, d, y, n = a
    f = fit(m, d, y, n)
    return f["bmdl"], f["ratio"]


def _rehearse_one(a):
    spec, dose, n, s = a
    rng = np.random.default_rng(s)
    y = rng.binomial(n.astype(int), true_logistic(*spec)(dose)).astype(float)
    fits, sel, rule = run_all(dose, y, n)
    j1, _ = verdicts(sel)
    j2 = j2v2(sel, dose, n, B=200, seed=s + 1)
    return {"J1": j1, "J2v2": j2["J2v2"], "cov": j2.get("boot_coverage"), "med": j2.get("boot_median_ratio"),
            "sel": None if sel is None else sel["model"]}


def rehearse_v2(spec, nseed, reps, seed):
    from multiprocessing import Pool
    dose, n = design(true_logistic(*spec), nseed)
    with Pool(30) as pool:
        res = pool.map(_rehearse_one, [(spec, dose, n, seed * 1000 + i) for i in range(reps)])
    cnt = lambda k, v: sum(1 for r in res if r[k] == v)
    return {"spec": spec, "nseed": nseed, "reps": reps, "doses": dose.tolist(),
            "true_bmd": true_bmd(true_logistic(*spec)),
            "J1": {v: cnt("J1", v) for v in ("rewritten", "not_rewritten")},
            "J2v2": {v: cnt("J2v2", v) for v in ("usable", "insufficient", "n/a")}, "rows": res}


if __name__ == "__main__":
    mode = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"wst2_{mode}.json"
    allm = tuple(MODELS)
    if mode == "rehearsal":
        spec = (780, 6.0)  # logistic ED50=780, slope 6 per 100 counts; true BMD10 ~ 743 > 600
        r = {"true_curve": "logistic ED50=780, slope 6/100 counts", "true_bmd": true_bmd(true_logistic(*spec)),
             "n20": simulate(spec, 20, 60, 1, allm), "n3": simulate(spec, 3, 60, 2, allm)}
    elif mode == "rehearsal_v2":
        r = {"i_steep_n20": rehearse_v2((780, 6.0), 20, 30, 21), "ii_steep_n3": rehearse_v2((780, 6.0), 3, 30, 22),
             "iii_low_n20": rehearse_v2((450, 6.0), 20, 30, 23)}
        for k in r: r[k]["summary_only"] = True
    elif mode == "real":
        dose, y, n = map(np.array, zip(*ZHENG))
        fits, sel, rule = run_all(dose, y, n)
        j1, j2 = verdicts(sel)
        r = {"data": ZHENG, "fits": fits, "selected": sel, "selection_rule": rule, "J1": j1, "J2": j2,
             "monotone_test": monotone_test(y, n),
             "J2v2": None,
             "zero_obs_upper": {"single_cell_0of3": 1 - 0.05 ** (1 / 3), "pooled_0of15": 1 - 0.05 ** (1 / 15)}}
    elif mode == "budget":
        spec = json.loads(sys.argv[3])  # {"model":..., "curve":"logistic", "ed50":..., "slope":...}
        cs = (spec["ed50"], spec["slope"])
        r = {"spec": spec, "true_bmd": true_bmd(true_logistic(*cs)),
             "runs": [simulate(cs, k, 500, 10 + k, ("logistic",)) for k in (3, 5, 10, 20)]}
    if mode == "real":
        from multiprocessing import Pool
        with Pool(30) as pool:
            r["J2v2"] = j2v2(r["selected"], dose, n, B=200, seed=7, pool=pool)
    json.dump(r, open(out, "w"), indent=1, default=float)
    if mode == "rehearsal_v2":
        print({k: (v["J1"], v["J2v2"]) for k, v in r.items()})
    print(json.dumps(r, indent=1, default=float)[:4000])
