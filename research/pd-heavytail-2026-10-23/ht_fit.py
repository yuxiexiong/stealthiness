"""CONTRACT_HT: is PD's "two incompatible taus" an artefact of the single-exponential kernel?

    python3 -I ht_fit.py <extract_jump.json.gz> <out_dir> [--rehearse]

Zero GPU. Re-uses analyze_zg.py's read-outs verbatim; the ONLY thing replaced is the
kernel inside drive().  theta (read-out 1) and p (read-out 2) absorb the kernel's overall
scale, so every family has exactly ONE free shape parameter -- the same as exp's tau.

--rehearse runs the two-sided offline rehearsal of contract 6.1 (worlds U and N) and the
S4 specificity check, and writes REHEARSAL_HT.json.  The contract forbids running on the
real data until that passes.
"""
import gzip, hashlib, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZG = os.path.join(HERE, "..", "..", "attribution-microscope", "supplement", "jump", "zg")
SUP = os.path.join(ZG, "..")

EXTRACT, OUT = sys.argv[1], sys.argv[2]
REHEARSE = "--rehearse" in sys.argv
os.makedirs(OUT, exist_ok=True)

E = json.load(gzip.open(EXTRACT)) if EXTRACT.endswith(".gz") else json.load(open(EXTRACT))
LP = json.load(open(os.path.join(SUP, "lp", "results_lp.json")))["runs"]
XO = json.load(open(os.path.join(SUP, "xo", "results_xo.json")))["runs"]

# jump.py holds WARM/TOTAL; import without putting its dir first on sys.path (-I already
# blocks cwd, but the supplement dir carries other modules we do not want shadowing).
sys.path.insert(0, SUP)
import jump as J  # noqa: E402

TMAX = J.TOTAL

# ---------------------------------------------------------------- grids (contract §2, frozen)
TAUS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500, 700, 1000, float("inf")]
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0]
HORIZONS = [20, 30, 50, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 1240]

# published reference numbers the run must reproduce (contract §6 S1)
REF = {"r1_loo_tau400": 23.9, "r1_nested": 30.4, "r1_count_loo": 33.8, "r1_const_loo": 38.4,
       "r2_rss_tau40": 665.0, "r2_r2_tau40": 0.840, "r2_r2_nopulse": 0.598}

LAGS = np.arange(TMAX + 1)


def kernel(family, shape):
    """K(Delta) on 0..TMAX, K(0)=1."""
    if family == "EXP":
        if math.isinf(shape):
            return np.ones(TMAX + 1)
        return np.exp(-LAGS / shape)
    if family == "PL":
        return (1.0 + LAGS) ** (-shape)
    if family == "LOG":
        k = 1.0 - np.log1p(LAGS) / math.log1p(shape)
        return np.clip(k, 0.0, None)
    raise ValueError(family)


FAMILIES = {"EXP": TAUS, "PL": ALPHAS, "LOG": HORIZONS}


def counts(steps):
    c = np.zeros(TMAX + 1)
    np.add.at(c, np.asarray(steps), 1)
    return c


def drive(steps, family, shape):
    """A(t) = sum_{t_i <= t} K(t - t_i), by direct convolution (any kernel)."""
    c = counts(steps)
    K = kernel(family, shape)
    return np.convolve(c, K)[: TMAX + 1]


# ---------------------------------------------------------------- read-out 1 (verbatim from analyze_zg)
def crossing(A, theta):
    idx = np.flatnonzero(A >= theta)
    return int(idx[0]) if len(idx) else TMAX + 200  # never: penalised


def fit_theta(drives, targets):
    cand = np.unique(np.concatenate([d[1:] for d in drives]))
    cand = cand[cand > 0]
    if len(cand) > 3000:
        cand = np.quantile(cand, np.linspace(0, 1, 3000))
    best, best_mae = None, None
    for th in cand:
        mae = float(np.mean([abs(crossing(d, th) - y) for d, y in zip(drives, targets)]))
        if best_mae is None or mae < best_mae:
            best, best_mae = float(th), mae
    return best, best_mae


def loo(names, ys, D):
    """plain LOO, theta refit per fold.  Returns (mean abs err, per-fold errors)."""
    errs = []
    for i in range(len(names)):
        tr = [j for j in range(len(names)) if j != i]
        th, _ = fit_theta([D[names[j]] for j in tr], [ys[j] for j in tr])
        errs.append(abs(crossing(D[names[i]], th) - ys[i]))
    return float(np.mean(errs)), errs


def nested_loo(names, ys, Dby, shapes):
    out = []
    for i in range(len(names)):
        inner = [j for j in range(len(names)) if j != i]
        sub = [names[j] for j in inner]
        suby = [ys[j] for j in inner]
        best = min(shapes, key=lambda s: loo(sub, suby, Dby[s])[0])
        th, _ = fit_theta([Dby[best][n] for n in sub], suby)
        out.append({"held_out": names[i], "shape": str(best),
                    "pred": crossing(Dby[best][names[i]], th), "obs": ys[i]})
    return float(np.mean([abs(o["pred"] - o["obs"]) for o in out])), out


# ---------------------------------------------------------------- read-out 2 (verbatim)
def joint_fit(rows_X, y, n_arms, arm_ids, arm_sched, D):
    X = np.zeros((len(rows_X), n_arms + 2))
    for r, (ai, t) in enumerate(rows_X):
        X[r, ai] = 1
        X[r, n_arms] = t / 100.0
        X[r, n_arms + 1] = D[arm_sched[arm_ids[r]]][t]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return {"g_per100": float(beta[n_arms]), "p": float(beta[n_arms + 1]),
            "r2": float(1 - resid.var() / y.var()), "rss": float((resid ** 2).sum())}


# ---------------------------------------------------------------- acceptance sets (contract §3)
F95_1_386 = 3.8653  # F_{0.95}(1, 386)
RSS_FACTOR = 1.0 + F95_1_386 / 386.0


def accept_r2(rss_by_shape):
    lo = min(rss_by_shape.values())
    return [s for s, v in rss_by_shape.items() if v <= lo * RSS_FACTOR], lo * RSS_FACTOR


def accept_r1(loo_by_shape, errs_by_shape):
    best = min(loo_by_shape, key=lambda s: loo_by_shape[s])
    e = np.asarray(errs_by_shape[best], float)
    se = float(e.std(ddof=1) / math.sqrt(len(e)))
    thr = loo_by_shape[best] + se
    return [s for s, v in loo_by_shape.items() if v <= thr], thr, se, best


def verdict(I1, I2, loo_by_shape):
    shared = [s for s in I1 if s in I2]
    if not shared:
        return "NOT_UNIFIED", shared, []
    ok = [s for s in shared if loo_by_shape[s] <= REF["r1_count_loo"]]
    return ("UNIFIED" if ok else "DEGENERATE"), shared, ok


# ================================================================ assemble the real data
def first_cross(steps, med):
    for k in range(1, len(steps)):
        if med[k - 1] < 0 <= med[k]:
            return steps[k - 1] + (0 - med[k - 1]) / (med[k] - med[k - 1]) * (steps[k] - steps[k - 1])
    return None


tm0 = {a: first_cross([p["step"] for p in pts], [p["median"] for p in pts]) for a, pts in E["lp"].items()}
selfcheck = {a: abs(tm0[a] - LP[a]["t_M0"]) < 1e-9 for a in LP}
assert all(selfcheck.values()), selfcheck
SELFCHECK = f"{sum(selfcheck.values())}/{len(selfcheck)}"

arr = {k: np.asarray(v) for k, v in E["arrivals"].items()}

# read-out 1 fit set: 11 distinct schedules (analyze_zg §v2)
xo_names = [a for a in E["lp"] if a.startswith("XO-")]
sp_names = sorted(a for a in E["lp"] if a.startswith("SP-"))
ps_names = sorted(a for a in E["lp"] if a.startswith("PS-"))
xo_sched = {a: "O" + a.split("-O")[1] for a in xo_names}
ps_sched = {a: "O1001" for a in ps_names}
arm_sched = {**xo_sched, **{a: a for a in sp_names}, **ps_sched}

o1001_arms = [a for a in xo_names if xo_sched[a] == "O1001"] + ["PS-P0"]
y_o1001 = float(np.mean([tm0[a] for a in o1001_arms]))
names1 = ["O1001"] + [xo_sched[a] for a in xo_names if xo_sched[a] != "O1001"] + sp_names
y1 = [y_o1001] + [tm0[a] for a in xo_names if xo_sched[a] != "O1001"] + [tm0[a] for a in sp_names]

# read-out 2 design matrix
rows_X, rows_y, arm_ids = [], [], []
for ai, a in enumerate(sorted(arm_sched)):
    for p in E["lp"][a]:
        if 40 <= p["step"] < tm0[a] - 10:
            rows_X.append((ai, p["step"]))
            rows_y.append(p["median"])
            arm_ids.append(a)
n_arms = len(arm_sched)
y2 = np.array(rows_y)
SCHEDS = sorted(set(arm_sched.values()))


def run_family(fam, shapes, targets1, y_r2):
    Dby = {s: {k: drive(arr[k], fam, s) for k in set(names1) | set(SCHEDS)} for s in shapes}
    loo_by, errs_by, theta_by = {}, {}, {}
    for s in shapes:
        th, _ = fit_theta([Dby[s][n] for n in names1], targets1)
        m, errs = loo(names1, targets1, Dby[s])
        loo_by[str(s)], errs_by[str(s)], theta_by[str(s)] = m, errs, th
    rss_by, imp_by = {}, {}
    for s in shapes:
        d = joint_fit(rows_X, y_r2, n_arms, arm_ids, arm_sched, Dby[s])
        rss_by[str(s)], imp_by[str(s)] = d["rss"], d
    I1, thr1, se1, best1 = accept_r1(loo_by, errs_by)
    I2, thr2 = accept_r2(rss_by)
    v, shared, ok = verdict(I1, I2, loo_by)
    return {"family": fam, "loo": loo_by, "theta": theta_by, "impulse": imp_by, "rss": rss_by,
            "I1": I1, "I1_threshold": thr1, "I1_se": se1, "I1_best": best1,
            "I2": I2, "I2_threshold": thr2, "I2_best": min(rss_by, key=lambda k: rss_by[k]),
            "shared": shared, "shared_passing_G1": ok, "verdict": v,
            "I1_coverage": len(I1) / len(shapes)}, Dby


# ================================================================ rehearsal (contract §6.1)
def synth_r1(fam, shape, theta_q, rng, noise=1.4):
    """crossing targets generated from a TRUE kernel on the real schedules."""
    D = {k: drive(arr[k], fam, shape) for k in names1}
    vals = np.concatenate([D[n][1:] for n in names1])
    theta = float(np.quantile(vals[vals > 0], theta_q))
    return [crossing(D[n], theta) + float(rng.normal(0, noise)) for n in names1], theta


def synth_r2(fam, shape, rng, p=1.0, g=2.13, resid_sd=1.3):
    """margin rows generated from a TRUE kernel, same design as the real read-out 2."""
    D = {k: drive(arr[k], fam, shape) for k in SCHEDS}
    m0 = {a: -18.0 + 0.3 * i for i, a in enumerate(sorted(arm_sched))}
    out = np.empty(len(rows_X))
    for r, (ai, t) in enumerate(rows_X):
        a = arm_ids[r]
        out[r] = m0[a] + g * t / 100.0 + p * D[arm_sched[a]][t] + rng.normal(0, resid_sd)
    return out


def rehearse():
    rng = np.random.default_rng(20261023)
    rep = {}
    # ---- world U: one true heavy-tail kernel behind BOTH read-outs
    tU1, thU = synth_r1("PL", 0.8, 0.55, rng)
    tU2 = synth_r2("PL", 0.8, rng)
    for fam, shapes in (("PL", ALPHAS), ("EXP", TAUS)):
        r, _ = run_family(fam, shapes, tU1, tU2)
        rep[f"U_{fam}"] = {k: r[k] for k in ("verdict", "I1", "I2", "shared", "shared_passing_G1",
                                            "I1_best", "I2_best", "I1_coverage")}
    # ---- world N: genuinely two timescales (read-out 1 slow, read-out 2 fast)
    tN1, thN = synth_r1("EXP", 400, 0.55, rng)
    tN2 = synth_r2("EXP", 40, rng)
    for fam, shapes in (("EXP", TAUS), ("PL", ALPHAS)):
        r, _ = run_family(fam, shapes, tN1, tN2)
        rep[f"N_{fam}"] = {k: r[k] for k in ("verdict", "I1", "I2", "shared", "shared_passing_G1",
                                            "I1_best", "I2_best", "I1_coverage")}
    rep["theta_used"] = {"U": thU, "N": thN}
    rep["gates"] = {
        "U_side_demonstrable": rep["U_PL"]["verdict"] == "UNIFIED",
        "N_side_demonstrable": rep["N_EXP"]["verdict"] == "NOT_UNIFIED",
        "S4_specificity_ok": rep["N_PL"]["verdict"] != "UNIFIED",
    }
    rep["gates"]["PASS"] = all(rep["gates"].values())
    return rep


# ================================================================ main
sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()

if REHEARSE:
    rep = rehearse()
    rep["code_sha256"] = sha
    json.dump(rep, open(os.path.join(OUT, "REHEARSAL_HT.json"), "w"), indent=1, default=str)
    g = rep["gates"]
    print("REHEARSAL", "PASS" if g["PASS"] else "FAIL")
    for k, v in g.items():
        print(f"  {k}: {v}")
    for w in ("U_PL", "U_EXP", "N_EXP", "N_PL"):
        print(f"  {w}: {rep[w]['verdict']}  I1={rep[w]['I1']}  I2={rep[w]['I2']}  shared={rep[w]['shared']}")
    sys.exit(0 if g["PASS"] else 3)

res = {"selfcheck_lp": SELFCHECK, "n_r2_points": int(len(y2)), "n_arms": n_arms,
       "r1_fit_set": names1, "r1_targets": y1, "code_sha256": sha, "ref": REF}
for fam, shapes in FAMILIES.items():
    r, _ = run_family(fam, shapes, y1, y2)
    res[fam] = r

# S1: reproduce the published exponential numbers
e = res["EXP"]
_, nested = nested_loo(names1, y1, {s: {k: drive(arr[k], "EXP", s) for k in set(names1)} for s in TAUS}, TAUS)
nested_mae = float(np.mean([abs(o["pred"] - o["obs"]) for o in nested]))
res["EXP"]["nested_loo_mae"] = nested_mae
res["EXP"]["nested"] = nested
s1 = {
    "r1_loo_tau400": (e["loo"]["400"], REF["r1_loo_tau400"], abs(e["loo"]["400"] - REF["r1_loo_tau400"]) <= 1.0),
    "r1_nested": (nested_mae, REF["r1_nested"], abs(nested_mae - REF["r1_nested"]) <= 1.0),
    "r2_rss_tau40": (e["rss"]["40"], REF["r2_rss_tau40"], abs(e["rss"]["40"] / REF["r2_rss_tau40"] - 1) <= 0.01),
    "r2_r2_tau40": (e["impulse"]["40"]["r2"], REF["r2_r2_tau40"], abs(e["impulse"]["40"]["r2"] - REF["r2_r2_tau40"]) <= 0.005),
}
res["S1"] = {k: {"got": v[0], "want": v[1], "pass": bool(v[2])} for k, v in s1.items()}
res["S1_PASS"] = all(v[2] for v in s1.values())
res["S2_no_range"] = {f: res[f]["I1_coverage"] >= 0.8 for f in FAMILIES}

json.dump(res, open(os.path.join(OUT, "RESULT_HT.json"), "w"), indent=1, default=str)

print("S1 reproduce published EXP numbers:", "PASS" if res["S1_PASS"] else "FAIL")
for k, v in res["S1"].items():
    print(f"  {k}: got {v['got']:.4g} want {v['want']:.4g} -> {'ok' if v['pass'] else 'FAIL'}")
print()
for fam in FAMILIES:
    r = res[fam]
    print(f"{fam}: verdict={r['verdict']}")
    print(f"  read-out 1 best={r['I1_best']} LOO={r['loo'][r['I1_best']]:.1f} thr={r['I1_threshold']:.1f} (SE {r['I1_se']:.1f})")
    print(f"     I1={r['I1']}  coverage={r['I1_coverage']:.0%}")
    print(f"  read-out 2 best={r['I2_best']} RSS={r['rss'][r['I2_best']]:.1f} thr={r['I2_threshold']:.1f}")
    print(f"     I2={r['I2']}")
    print(f"  shared={r['shared']}  passing G1={r['shared_passing_G1']}")
