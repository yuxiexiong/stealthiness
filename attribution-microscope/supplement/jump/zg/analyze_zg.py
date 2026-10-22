"""Zero-GPU analysis for the jump line (2026-10-22, exploratory, NOT a contract).

    python analyze_zg.py <extract.json> <out_dir>

Inputs: the CPU extract from the server (poison arrival steps for 9 original orders,
6 SP splices, PS-P4; LP read-outs and behavioral ASR for 19 arms) plus the frozen
result files of XO / SP / LP / PS / r1 in the parent directories.

Outputs: results_zg.json (all numbers quoted in RESULTS_ZG.md), fig_data.json
(the three panels of the opener figure).

Model (pulse-decay, "PD"): every poison row consumed at step t_i adds a unit pulse
that decays with time constant tau; drive A(t) = sum_{t_i <= t} w_i exp(-(t-t_i)/tau);
predicted take-off = first step with A(t) >= theta. tau = inf is the cumulative-count
model (r2's G). Variant "PD-lr" weights each pulse by the lr multiplier lam(t_i).
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jump as J  # lam(t), WARM, TOTAL

HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
T50_20 = {1001: 360, 1007: 470, 1008: 410, 1011: 410, 1009: 330, 1004: 370, 1005: 390, 1006: 390, 1010: 350}  # sp.T50, 20-step grid
LP = json.load(open(os.path.join(HERE, "..", "lp", "results_lp.json")))["runs"]
XO = json.load(open(os.path.join(HERE, "..", "xo", "results_xo.json")))["runs"]
R1B = json.load(open(os.path.join(HERE, "..", "results.json")))["B"]
TMAX = J.TOTAL
TAUS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500, 700, 1000, float("inf")]
LAM = np.array([J.lam(t) for t in range(TMAX + 1)])


# ---------------------------------------------------------------- read-outs
def first_cross(steps, med):
    """first zero crossing of the median core margin, linear interpolation (refs_sf rule)."""
    for k in range(1, len(steps)):
        if med[k - 1] < 0 <= med[k]:
            return steps[k - 1] + (0 - med[k - 1]) / (med[k] - med[k - 1]) * (steps[k] - steps[k - 1])
    return None


tm0 = {}
for arm, pts in E["lp"].items():
    tm0[arm] = first_cross([p["step"] for p in pts], [p["median"] for p in pts])
selfcheck = {a: (tm0[a], LP[a]["t_M0"], abs(tm0[a] - LP[a]["t_M0"]) < 1e-9) for a in LP}
assert all(v[2] for v in selfcheck.values()), selfcheck

# ---------------------------------------------------------------- schedules
arr = {k: np.asarray(v) for k, v in E["arrivals"].items()}


def counts(steps):
    c = np.zeros(TMAX + 1); np.add.at(c, np.asarray(steps), 1); return c


def drive(steps, tau, lr_weight=False):
    c = counts(steps)
    if lr_weight: c = c * LAM
    if math.isinf(tau): return np.cumsum(c)
    A = np.zeros(TMAX + 1); d = math.exp(-1.0 / tau)
    for t in range(1, TMAX + 1): A[t] = A[t - 1] * d + c[t]
    return A


def crossing(A, theta):
    idx = np.flatnonzero(A >= theta)
    return int(idx[0]) if len(idx) else TMAX + 200  # never: penalised


def fit_theta(drives, targets):
    """theta minimising MAE of predicted crossing on the fit set (scan over all drive values)."""
    cand = np.unique(np.concatenate([d[1:] for d in drives]))
    cand = cand[(cand > 0)]
    if len(cand) > 3000: cand = np.quantile(cand, np.linspace(0, 1, 3000))
    best = (float("inf"), None)
    for th in cand:
        mae = np.mean([abs(crossing(d, th) - y) for d, y in zip(drives, targets)])
        if mae < best[0]: best = (mae, float(th))
    return best[1], best[0]


orig = sorted(T50_20)  # 9 original orders
fit_names = [f"O{s}" for s in orig]; fit_y = [T50_20[s] for s in orig]
sp_names = [a for a in sorted(LP) if a.startswith("SP-")]; sp_y = [LP[a]["t_M0"] for a in sp_names]
xo_names = [a for a in sorted(LP) if a.startswith("XO-")]; xo_y = [LP[a]["t_M0"] for a in xo_names]
xo_sched = {a: f"O{[r for r in XO if r['arm'] == a][0]['order']}" for a in xo_names}
ps_names = ["PS-P0", "PS-P1", "PS-P2", "PS-P3", "PS-P4"]; ps_y = [tm0[a] for a in ps_names]
ps_sched = {a: ("PS-P4" if a == "PS-P4" else "O1001") for a in ps_names}


def evaluate(lr_weight):
    res = {}
    for tau in TAUS:
        D = {k: drive(v, tau, lr_weight) for k, v in arr.items()}
        th, mae_fit = fit_theta([D[n] for n in fit_names], fit_y)
        # leave-one-out over the 9 original orders
        loo = []
        for i in range(len(fit_names)):
            tr = [j for j in range(len(fit_names)) if j != i]
            th_i, _ = fit_theta([D[fit_names[j]] for j in tr], [fit_y[j] for j in tr])
            loo.append(abs(crossing(D[fit_names[i]], th_i) - fit_y[i]))
        pred_sp = [crossing(D[a], th) for a in sp_names]
        pred_xo = [crossing(D[xo_sched[a]], th) for a in xo_names]
        pred_ps = [crossing(D[ps_sched[a]], th) for a in ps_names]
        res[str(tau)] = {"theta": th, "mae_fit": mae_fit, "loo_mae": float(np.mean(loo)), "loo_abs": loo,
                         "pred_fit": {n: crossing(D[n], th) for n in fit_names},
                         "sp": {"pred": dict(zip(sp_names, pred_sp)), "mae": float(np.mean(np.abs(np.array(pred_sp) - sp_y)))},
                         "xo": {"pred": dict(zip(xo_names, pred_xo)), "mae": float(np.mean(np.abs(np.array(pred_xo) - xo_y)))},
                         "ps": {"pred": dict(zip(ps_names, pred_ps)), "mae": float(np.mean(np.abs(np.array(pred_ps) - ps_y)))}}
    return res


const_fit = float(np.mean(fit_y))
baselines = {"const_loo_mae": float(np.mean([abs(np.mean([fit_y[j] for j in range(9) if j != i]) - fit_y[i]) for i in range(9)])),
             "const_sp_mae": float(np.mean(np.abs(np.array(sp_y) - const_fit))),
             "const_xo_mae": float(np.mean(np.abs(np.array(xo_y) - const_fit))),
             "const_ps_mae": float(np.mean(np.abs(np.array(ps_y) - const_fit))),
             "const_fit_mean": const_fit}
PD = evaluate(False); PDlr = evaluate(True)


def best_tau(res, key):
    return min(res, key=lambda k: res[k][key])


# ---------------------------------------------------------------- impulse response from the margin curves
# M_a(t) = m0_a + g t + p A_tau(t) on pre-crossing checkpoints of all 19 arms (joint least squares per tau)
arm_sched = {**{a: xo_sched[a] for a in xo_names}, **{a: a for a in sp_names}, **ps_sched}
rows_X, rows_y, arm_ids = [], [], []
for ai, a in enumerate(sorted(arm_sched)):
    for p in E["lp"][a]:
        if p["step"] < tm0[a] - 10 and p["step"] >= 40:   # after warm-up transient, before crossing
            rows_X.append((ai, p["step"])); rows_y.append(p["median"]); arm_ids.append(a)
n_arms = len(arm_sched); y = np.array(rows_y)
imp = {}
for tau in TAUS:
    D = {k: drive(arr[k], tau) for k in set(arm_sched.values())}
    X = np.zeros((len(rows_X), n_arms + 2))
    for r, (ai, t) in enumerate(rows_X):
        X[r, ai] = 1; X[r, n_arms] = t / 100.0; X[r, n_arms + 1] = D[arm_sched[arm_ids[r]]][t]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; r2 = 1 - resid.var() / y.var()
    imp[str(tau)] = {"g_per100": float(beta[n_arms]), "p": float(beta[n_arms + 1]), "r2": float(r2), "rss": float((resid ** 2).sum())}
X0 = np.zeros((len(rows_X), n_arms + 1))
for r, (ai, t) in enumerate(rows_X): X0[r, ai] = 1; X0[r, n_arms] = t / 100.0
b0, *_ = np.linalg.lstsq(X0, y, rcond=None); r0 = y - X0 @ b0
imp["no_pulse"] = {"g_per100": float(b0[n_arms]), "r2": float(1 - r0.var() / y.var()), "rss": float((r0 ** 2).sum()), "n": int(len(y))}

# ---------------------------------------------------------------- 1-step descriptive (r1 B, order 1001 = s1; s2 = seed 1002)
c1001 = counts(arr["O1001"]); c1002 = counts(arr["O1002"])
one_step = {}
for s, c in (("s1", c1001), ("s2", c1002)):
    b = R1B[s]; rows = []
    for k in range(1, len(b["steps"])):
        t0, t1 = b["steps"][k - 1], b["steps"][k]
        rows.append({"from": t0, "to": t1, "arrivals": int(c[t0 + 1:t1 + 1].sum()), "dT3": b["median_T3"][k] - b["median_T3"][k - 1], "dT3_per_step": (b["median_T3"][k] - b["median_T3"][k - 1]) / (t1 - t0)})
    one_step[s] = rows

# ---------------------------------------------------------------- ladder predictions (best PD by LOO)
bt = best_tau(PD, "loo_mae"); tau_b = float(bt); th_b = PD[bt]["theta"]
ladder = {}
for s in (1001, 1007, 1009):
    base = arr[f"O{s}"]; D0 = drive(base, tau_b); t0 = crossing(D0, th_b); out = {"t_pred": t0}
    for name, lo, hi, shift in (("late_window_+30", t0 - 50, t0, 30), ("early_window_+30", 50, 100, 30), ("late_window_-30", t0 - 50, t0, -30), ("early_window_-30", 50, 100, -30)):
        m = (base >= lo) & (base <= hi); new = base.copy(); new[m] = np.clip(new[m] + shift, 1, TMAX)
        out[name] = {"n_rows_moved": int(m.sum()), "t_pred_new": crossing(drive(new, tau_b), th_b)}
    ladder[f"O{s}"] = out

# ---------------------------------------------------------------- figure data
figA = {}
for s in orig:
    c = np.cumsum(counts(arr[f"O{s}"])); figA[str(s)] = {"K": c[:601].astype(int).tolist(), "t50": T50_20[s], "K_at_t50": int(c[T50_20[s]])}
figB = [{"arm": r["arm"], "init": r["init"], "order": r["order"], "t50": r["t50"], "t_M0": LP[r["arm"]]["t_M0"]} for r in XO]
figC = {a: [(p["step"], p["asr"]) for p in E["behavioral"][a] if 300 <= p["step"] <= 420] for a in ps_names}
json.dump({"A": figA, "B": figB, "C": figC}, open(os.path.join(OUT, "fig_data.json"), "w"))
json.dump({"selfcheck_tM0": {a: v[0] for a, v in selfcheck.items()}, "tM0_all": tm0, "baselines": baselines,
           "PD": PD, "PD_lr": PDlr, "best_tau_loo": {"PD": bt, "PD_lr": best_tau(PDlr, "loo_mae")},
           "impulse": imp, "one_step": one_step, "ladder": {"tau": tau_b, "theta": th_b, "orders": ladder},
           "K_at_t50": {str(s): figA[str(s)]["K_at_t50"] for s in orig},
           "n_pre_crossing_points": int(len(y))}, open(os.path.join(OUT, "results_zg.json"), "w"), indent=1)
print("selfcheck 14/14 ok; best tau (LOO) PD", bt, PD[bt]["loo_mae"], "const", baselines["const_loo_mae"])
for k in PD: print(f"tau={k:>5} theta={PD[k]['theta']:.2f} fitMAE={PD[k]['mae_fit']:.1f} LOO={PD[k]['loo_mae']:.1f} SP={PD[k]['sp']['mae']:.1f} XO={PD[k]['xo']['mae']:.1f} PS={PD[k]['ps']['mae']:.1f}")
print("baselines", baselines)
print("impulse", {k: (round(v['r2'], 4), round(v.get('p', 0), 3), round(v['g_per100'], 3)) for k, v in imp.items()})
print("ladder", json.dumps(ladder))

# ================================================================ v2: consistent read-outs, nested LOO
# Fit set = 11 distinct schedules all read on the LP 10-step grid (first zero crossing of the median core margin):
# O1001 (mean over the 5 runs with different inits: XO x4 + PS-P0), O1007/O1008/O1009/O1011 (XO-I1001-O*), 6 SP splices.
sched_y = {"O1001": float(np.mean([tm0[a] for a in xo_names if xo_sched[a] == "O1001"] + [tm0["PS-P0"]]))}
for a in xo_names:
    if xo_sched[a] != "O1001": sched_y[xo_sched[a]] = tm0[a]
for a in sp_names: sched_y[a] = tm0[a]
names2 = sorted(sched_y); y2 = [sched_y[n] for n in names2]
D_by_tau = {tau: {k: drive(arr[k], tau) for k in arr} for tau in TAUS}


def loo_mae(names, ys, tau, idx):
    """plain LOO over idx (list of positions) with theta refit each fold."""
    errs = []
    for i in idx:
        tr = [j for j in idx if j != i]
        th, _ = fit_theta([D_by_tau[tau][names[j]] for j in tr], [ys[j] for j in tr])
        errs.append(abs(crossing(D_by_tau[tau][names[i]], th) - ys[i]))
    return float(np.mean(errs)), errs


fixed = {}
for tau in TAUS:
    th, mf = fit_theta([D_by_tau[tau][n] for n in names2], y2)
    m, errs = loo_mae(names2, y2, tau, list(range(len(names2))))
    fixed[str(tau)] = {"theta": th, "mae_fit": mf, "loo_mae": m, "pred": {n: crossing(D_by_tau[tau][n], th) for n in names2}}
nested = []
for i in range(len(names2)):
    inner = [j for j in range(len(names2)) if j != i]
    best = min(TAUS, key=lambda tau: loo_mae(names2, y2, tau, inner)[0])
    th, _ = fit_theta([D_by_tau[best][names2[j]] for j in inner], [y2[j] for j in inner])
    nested.append({"held_out": names2[i], "tau": best, "pred": crossing(D_by_tau[best][names2[i]], th), "obs": y2[i]})
nested_mae = float(np.mean([abs(r["pred"] - r["obs"]) for r in nested]))
const2 = float(np.mean([abs(np.mean([y2[j] for j in range(len(y2)) if j != i]) - y2[i]) for i in range(len(y2))]))
# secondary: the four orders that only have a 20-step ASR t50 (grid caveat: 20-step t50 runs 0-30 steps later than the LP crossing)
best_all = min(TAUS, key=lambda tau: fixed[str(tau)]["loo_mae"]); th_all = fixed[str(best_all)]["theta"]
sec = {f"O{s}": {"pred": crossing(D_by_tau[best_all][f"O{s}"], th_all), "t50_20grid": T50_20[s]} for s in (1004, 1005, 1006, 1010)}
sec_mae = float(np.mean([abs(v["pred"] - v["t50_20grid"]) for v in sec.values()]))
sec_const = float(np.mean([abs(np.mean(y2) - v["t50_20grid"]) for v in sec.values()]))
ladder50 = {}
for s in (1001, 1007, 1009):
    base = arr[f"O{s}"]; th50 = fixed["50"]["theta"]; t0 = crossing(drive(base, 50), th50); out = {"t_pred": t0}
    for name, lo, hi, shift in (("late_window_+30", t0 - 50, t0, 30), ("early_window_+30", 50, 100, 30)):
        m = (base >= lo) & (base <= hi); new = base.copy(); new[m] = np.clip(new[m] + shift, 1, TMAX)
        out[name] = {"n_rows_moved": int(m.sum()), "t_pred_new": crossing(drive(new, 50), th50)}
    ladder50[f"O{s}"] = out
R = json.load(open(os.path.join(OUT, "results_zg.json")))
R["v2"] = {"fit_set": sched_y, "fixed_tau": fixed, "nested_loo": nested, "nested_loo_mae": nested_mae, "const_loo_mae": const2,
           "best_tau_plain_loo": best_all, "secondary_20grid": {"pred": sec, "mae": sec_mae, "const_mae": sec_const}, "ladder_tau50": ladder50}
json.dump(R, open(os.path.join(OUT, "results_zg.json"), "w"), indent=1)
print("\n== v2 fit set", {k: round(v, 1) for k, v in sched_y.items()})
for k in fixed: print(f"tau={k:>5} theta={fixed[k]['theta']:.2f} fitMAE={fixed[k]['mae_fit']:.1f} LOO={fixed[k]['loo_mae']:.1f}")
print("nested LOO MAE", round(nested_mae, 1), "const LOO", round(const2, 1), [(r['held_out'], r['tau'], r['pred'], round(r['obs'], 1)) for r in nested])
print("secondary (20-grid orders) tau", best_all, sec, "mae", round(sec_mae, 1), "const", round(sec_const, 1))
print("ladder tau50", ladder50)
print("one_step", json.dumps(R["one_step"]))
print("n pre-crossing points", R["n_pre_crossing_points"], {k: (round(v["rss"], 1)) for k, v in R["impulse"].items()})

# ================================================================ v3: margin model "MD" (shared m0 + g t + p A_tau), crossing predicted from the margin fit
# fit (m0, g, p) by least squares on the pre-crossing margin points of the training arms (one shared intercept),
# predict the held-out schedule's crossing as the first t with m0 + g t/100 + p A_tau(t) >= 0.
arm_pts = {a: [(p["step"], p["median"]) for p in E["lp"][a] if 40 <= p["step"] < tm0[a] - 10] for a in arm_sched}
sched_arms = {}
for a, s in arm_sched.items(): sched_arms.setdefault(s, []).append(a)
sched_arms["O1001"] = [a for a in sched_arms["O1001"] if a != "PS-P4"]  # P4 has its own schedule; keep P0-P3 + XO x4 as O1001


def md_fit(train_scheds, tau):
    X, Y = [], []
    for s in train_scheds:
        for a in sched_arms[s]:
            for t, m in arm_pts[a]: X.append([1.0, t / 100.0, D_by_tau[tau][s][t]]); Y.append(m)
    X, Y = np.array(X), np.array(Y); b, *_ = np.linalg.lstsq(X, Y, rcond=None); return b, float(((Y - X @ b) ** 2).sum()), len(Y)


def md_cross(b, s, tau):
    t = np.arange(TMAX + 1); f = b[0] + b[1] * t / 100.0 + b[2] * D_by_tau[tau][s]
    idx = np.flatnonzero((f >= 0) & (t >= 40)); return int(idx[0]) if len(idx) else TMAX + 200


md = {}
for tau in TAUS:
    b, rss, n = md_fit(names2, tau)
    errs = []
    for i, s in enumerate(names2):
        bi, _, _ = md_fit([x for x in names2 if x != s], tau)
        errs.append(abs(md_cross(bi, s, tau) - y2[i]))
    md[str(tau)] = {"m0": float(b[0]), "g_per100": float(b[1]), "p": float(b[2]), "rss": rss, "n": n, "loo_mae": float(np.mean(errs)), "loo_errs": errs,
                    "pred_in_sample": {s: md_cross(b, s, tau) for s in names2}}
R = json.load(open(os.path.join(OUT, "results_zg.json"))); R["v3_md"] = md
json.dump(R, open(os.path.join(OUT, "results_zg.json"), "w"), indent=1)
print("\n== v3 MD (shared m0 + drift + pulses), LOO crossing MAE by tau; const", round(const2, 1))
for k in md: print(f"tau={k:>5} m0={md[k]['m0']:.2f} g/100={md[k]['g_per100']:.2f} p={md[k]['p']:.3f} rss={md[k]['rss']:.0f} LOO={md[k]['loo_mae']:.1f}")
k = min(md, key=lambda k: md[k]["loo_mae"]); print("best", k, {s: (md[k]['pred_in_sample'][s], round(sched_y[s], 1)) for s in names2})
