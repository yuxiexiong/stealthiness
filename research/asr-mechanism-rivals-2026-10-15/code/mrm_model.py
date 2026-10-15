"""Minimal readout model (CONTRACT_MRM.md). numpy only, read-only, zero GPU.

Two independent halves, each falsifiable.

WIDTH  ASR(t) = fraction of probes with c(t) + theta_i > 0, so the 0.1 -> 0.9 window is the time c takes to sweep
       the 10-90% quantile band of theta. theta is read two ways: BASE (pre-poison margins at step 0; the LoRA
       adapter starts at zero, so for language runs step 0 is the base model) and WIN (atlas additive offsets on
       [p, b]). Three width predictors against the observed width b - a:
         W1 = band_win  / v_in      structure check (close to an identity)
         W2 = band_base / v_in      is the band already set before training?
         W3 = band_base / v_pre20   full forecast from pre-window quantities only
       W3 is compared with a leave-one-out null (family median of the other runs' widths).

ONSET  Leaky poison accumulator x(t) = sum over updates u <= t of lr(u) * n_poison(u) * exp(-(t - u) / tau).
       One shared tau is chosen on fit seeds as the value that makes x(t50) most alike (minimum coefficient of
       variation); tau = inf is the plain LR-weighted count. A run's onset is predicted as the first update where
       x reaches the fit seeds' median x(t50). Nulls: mean onset of the fit seeds; tau = inf; and the same tau and
       threshold applied to another seed's poison schedule (random derangements).

    python mrm_model.py --base /workspace/cross-model-asr --out results/mrm.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl
from b0res_squad import DENSE as SQ_DENSE, ORIG as SQ_ORIG, merge
from fixedpre import ratio_fixed
from r_corpus import SERVER

LL_ORIG = {s: f"20261007_vim_llama_v1/llama/runs/formal_s{s}_poison" for s in range(1001, 1006)}
LL_DENSE = {s: f"20261007_vim_llama_v1/llama/refinements/replay_s{s}_poison" for s in range(1001, 1006)}
GATE_TOL = 0.02
TAUS = [5, 10, 20, 40, 80, 160, 320, 640, math.inf]
Q_LO, Q_HI = 0.10, 0.90
L_PRE = 20
N_SWAP = 2000
# --- frozen tiers (CONTRACT_MRM.md section 4) ---
STRUCT_OK = 0.5        # median |log2 W1/obs|
PRESET_GAP = 0.25      # median e2 - median e1
PRESET_MAX = 0.5       # median e2
FORMED_MIN = 1.0       # median e2
FORECAST_OK = 0.5      # median e3
FORECAST_BAD = 1.0
GAIN = 0.7             # model MAE must be <= 0.7 x the better of the mean and count nulls
SWAP_P, SWAP_P_NO = 0.05, 0.2
CV_GAIN = 0.8          # finite tau must reach CV <= 0.8 x CV(inf)


# ---------------------------------------------------------------- loading
def poison_lr(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "training.jsonl")) if l.strip()]
    pois = {r["optimizer_step"]: int(r["poison_examples"]) for r in rows}
    lrs = {r["optimizer_step"]: float(r["lr"]) for r in rows}
    return pois, lrs


def step0_margins(run_dir, ids):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "samples-0000.jsonl")) if l.strip()]
    at = {r["id"]: r["trigger"]["margin"] for r in rows}
    if not all(i in at for i in ids):
        return None
    return np.array([at[i] for i in ids], float)


def load_language(base, orig, dense):
    """Merged 1-step trajectories (same gate as B0RES) plus step-0 base margins, poison counts and LRs."""
    runs = []
    for s in sorted(orig):
        do, dd = os.path.join(base, orig[s]), os.path.join(base, dense[s])
        o, d = rl.language_run(do), rl.language_run(dd)
        if o is None or d is None:
            runs.append({"seed": s, "status": "unreadable"})
            continue
        steps, X, F, diff = merge(o[:4], d[:4])
        if diff > GATE_TOL:
            runs.append({"seed": s, "status": "gate_failed", "gate_diff": diff})
            continue
        bm = step0_margins(do, o[3])
        pois, lrs = poison_lr(do)
        runs.append({"seed": s, "status": "ok", "steps": steps, "X": X, "F": F, "base_margin": bm,
                     "pois": pois, "lrs": lrs, "n_updates": max(pois)})
    return runs


def load_classification(base, fam):
    roots, pattern, _ = SERVER[fam]
    runs = []
    for d in rl.find_runs(base, roots, pattern):
        if "pilot" in os.path.basename(d):
            continue
        got = rl.classification_run(d)
        if got is None:
            continue
        steps, X, F = got[:3]
        bm = X[0].copy() if steps[0] == 0 else None
        runs.append({"seed": os.path.basename(d), "status": "ok", "steps": list(steps), "X": X, "F": F,
                     "base_margin": bm})
    return runs


# ---------------------------------------------------------------- WIDTH
def band(theta):
    return float(np.quantile(theta, Q_HI) - np.quantile(theta, Q_LO))


def lg(pred, obs):
    return float(abs(np.log2(pred / obs))) if (pred is not None and pred > 0 and obs > 0) else None


def width_record(steps, X, F, base_margin):
    steps = list(steps)
    asr, c = F.mean(1), X.mean(1)
    w = rc.window(steps, asr)
    if not w:
        return {"status": "no_window"}
    a, b = w
    ar = rc.atlas_ratio(steps, c, a, b)
    p = ar["p"] if ar else steps[0]
    ip, ia, ib = steps.index(p), steps.index(a), steps.index(b)
    theta_win = rc.offsets(X[ip:ib + 1])
    obs = b - a
    v_in = float((c[ib] - c[ia]) / obs)
    rf = ratio_fixed(steps, c, a, b, L_PRE)
    v_pre = rf["v_pre"] if rf else None
    out = {"status": "ok", "window": [a, b], "p": p, "obs_width": obs, "v_in": v_in, "v_pre20": v_pre,
           "band_win": band(theta_win)}
    if not v_in > 0:
        out["status"] = "nonpositive_v_in"
        return out
    out["W1"] = out["band_win"] / v_in
    out["e1"] = lg(out["W1"], obs)
    if base_margin is not None:
        theta_base = base_margin - base_margin.mean()
        out["band_base"] = band(theta_base)
        out["rank_base_win"] = rc.spearman(theta_base, theta_win)
        out["W2"] = out["band_base"] / v_in
        out["e2"] = lg(out["W2"], obs)
        if v_pre is not None and v_pre > 0:
            out["W3"] = out["band_base"] / v_pre
            out["e3"] = lg(out["W3"], obs)
    return out


def width_family(recs):
    ok = [r for r in recs if r.get("status") == "ok"]
    widths = [r["obs_width"] for r in ok]
    for k, r in enumerate(ok):          # leave-one-out null for W3: median width of the other runs
        others = widths[:k] + widths[k + 1:]
        r["e_null"] = lg(float(np.median(others)), r["obs_width"]) if others else None
    m = lambda key: rc.med([r.get(key) for r in ok if r.get(key) is not None])
    s = {"n": len(ok), "median_e1": m("e1"), "median_e2": m("e2"), "median_e3": m("e3"), "median_e_null": m("e_null"),
         "median_band_win": m("band_win"), "median_band_base": m("band_base"),
         "median_rank_base_win": m("rank_base_win"), "n_e3": sum(r.get("e3") is not None for r in ok)}
    s["tier_structure"] = "unmeasurable" if s["median_e1"] is None else (
        "structure_ok" if s["median_e1"] <= STRUCT_OK else "structure_fail")
    s["tier_band"] = tier_band(s["median_e1"], s["median_e2"])
    s["tier_forecast"] = tier_forecast(s["median_e3"], s["median_e_null"])
    return s


def tier_band(e1, e2):
    if e1 is None or e2 is None:
        return "unmeasurable"
    if e2 - e1 <= PRESET_GAP and e2 <= PRESET_MAX:
        return "band_preset"
    if e2 >= FORMED_MIN:
        return "band_formed"
    return "partial"


def tier_forecast(e3, enull):
    if e3 is None or enull is None:
        return "unmeasurable"
    if e3 <= FORECAST_OK and e3 < enull:
        return "forecastable"
    if e3 >= FORECAST_BAD or e3 >= enull:
        return "not_forecastable"
    return "partial"


# ---------------------------------------------------------------- ONSET
def accumulator(pois, lrs, tau, upto):
    """x[t] for t = 0..upto; update u moves x[u-1] -> x[u]."""
    x = np.zeros(upto + 1)
    decay = 0.0 if tau == 0 else (1.0 if tau == math.inf else math.exp(-1.0 / tau))
    for t in range(1, upto + 1):
        x[t] = x[t - 1] * decay + lrs.get(t, 0.0) * pois.get(t, 0)
    return x


def onset_fit(runs, taus):
    """Per tau: coefficient of variation of x(t50) across runs and the median x(t50) (the threshold)."""
    table, best = [], None
    for tau in taus:
        vals = np.array([accumulator(r["pois"], r["lrs"], tau, r["t50"])[r["t50"]] for r in runs])
        cv = float(vals.std() / vals.mean()) if vals.mean() > 0 else math.inf
        row = {"tau": tau, "cv": cv, "threshold": float(np.median(vals))}
        table.append(row)
        if best is None or cv < best["cv"]:
            best = row
    return best, table


def predict_one(pois, lrs, n_updates, tau, thr):
    """First update where x >= thr; a run that never reaches it is predicted at its last update (censored)."""
    x = accumulator(pois, lrs, tau, n_updates)
    hit = np.nonzero(x[1:] >= thr * (1 - 1e-9))[0]      # relative tolerance for float ties
    return (int(hit[0]) + 1, False) if len(hit) else (n_updates, True)


def mae(runs, tau, thr, schedules=None):
    """Mean |pred - t50|; schedules[k] (optional) replaces run k's own poison schedule."""
    errs, cens = [], 0
    for k, r in enumerate(runs):
        pois = r["pois"] if schedules is None else schedules[k]
        pred, c = predict_one(pois, r["lrs"], r["n_updates"], tau, thr)
        errs.append(abs(pred - r["t50"]))
        cens += c
    return float(np.mean(errs)), cens


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def swap_p(runs, pool, tau, thr, observed, rng):
    """Fraction of random schedule swaps (each run gets a different seed's schedule from pool) with MAE <= observed.
    pool = list of runs whose schedules may be borrowed; runs[k] must be pool[k] for k < len(runs) when pool == runs."""
    n = len(pool)
    if n < 2:
        return None
    cnt = 0
    for _ in range(N_SWAP):
        perm = derangement(n, rng)
        sched = [pool[perm[k]]["pois"] for k in range(len(runs))]
        cnt += mae(runs, tau, thr, sched)[0] <= observed
    return float((cnt + 1) / (N_SWAP + 1))


def loo(runs, taus, rng):
    """Leave-one-out within a family: fit tau and threshold on the others, predict the held-out run."""
    rows = []
    for k, r in enumerate(runs):
        train = runs[:k] + runs[k + 1:]
        best, table = onset_fit(train, taus)
        inf_row = next(t for t in table if t["tau"] == math.inf)
        pm, cm = predict_one(r["pois"], r["lrs"], r["n_updates"], best["tau"], best["threshold"])
        pi, ci = predict_one(r["pois"], r["lrs"], r["n_updates"], math.inf, inf_row["threshold"])
        rows.append({"seed": r["seed"], "t50": r["t50"], "tau": best["tau"], "thr": best["threshold"],
                     "pred": pm, "censored": cm,
                     "err": abs(pm - r["t50"]), "pred_count": pi, "err_count": abs(pi - r["t50"]),
                     "err_mean": abs(r["t50"] - float(np.mean([x["t50"] for x in train])))})
    out = {"runs": rows, "mae_model": float(np.mean([x["err"] for x in rows])),
           "mae_count": float(np.mean([x["err_count"] for x in rows])),
           "mae_mean": float(np.mean([x["err_mean"] for x in rows])),
           "n_censored": sum(x["censored"] for x in rows),
           "taus_chosen": [x["tau"] for x in rows]}
    # swap null on the LOO predictions: each fold's tau/threshold applied to another seed's schedule
    cnt = 0
    n = len(runs)
    for _ in range(N_SWAP):
        perm = derangement(n, rng)
        e = []
        for k, r in enumerate(runs):
            f = rows[k]
            pred, _ = predict_one(runs[perm[k]]["pois"], r["lrs"], r["n_updates"], f["tau"], f["thr"])
            e.append(abs(pred - r["t50"]))
        cnt += float(np.mean(e)) <= out["mae_model"]
    out["swap_p"] = float((cnt + 1) / (N_SWAP + 1))
    out["tier"] = tier_onset(out["mae_model"], out["mae_mean"], out["mae_count"], out["swap_p"])
    return out


def tier_onset(m, m_mean, m_count, p):
    if m >= m_mean or (p is not None and p >= SWAP_P_NO):
        return "no_accumulator"
    if m <= GAIN * min(m_mean, m_count) and p is not None and p < SWAP_P:
        return "accumulator_predicts"
    return "partial"


def held_out(fit, test, taus, rng):
    """tau from the fit family; threshold (H2, primary) refit leave-one-out inside the test family with tau fixed,
    and (H1, strict) transferred unchanged from the fit family."""
    best, table = onset_fit(fit, taus)
    inf_row = next(t for t in table if t["tau"] == math.inf)
    res = {"tau": best["tau"], "cv_best": best["cv"], "cv_inf": inf_row["cv"],
           "finite_tau_wins": bool(best["tau"] != math.inf and best["cv"] <= CV_GAIN * inf_row["cv"]),
           "fit_table": table}
    # H1 strict transfer
    m1, c1 = mae(test, best["tau"], best["threshold"])
    mi1, _ = mae(test, math.inf, inf_row["threshold"])
    mean_fit = float(np.mean([r["t50"] for r in fit]))
    mm1 = float(np.mean([abs(r["t50"] - mean_fit) for r in test]))
    res["H1"] = {"mae_model": m1, "mae_count": mi1, "mae_mean": mm1, "n_censored": c1,
                 "swap_p": swap_p(test, test, best["tau"], best["threshold"], m1, rng)}
    res["H1"]["tier"] = tier_onset(m1, mm1, mi1, res["H1"]["swap_p"])
    # H2 tau transferred, threshold leave-one-out within the test family
    rows = []
    for k, r in enumerate(test):
        tr = test[:k] + test[k + 1:]
        thr = onset_fit(tr, [best["tau"]])[0]["threshold"]
        thr_inf = onset_fit(tr, [math.inf])[0]["threshold"]
        pm, cm = predict_one(r["pois"], r["lrs"], r["n_updates"], best["tau"], thr)
        pi, _ = predict_one(r["pois"], r["lrs"], r["n_updates"], math.inf, thr_inf)
        rows.append({"seed": r["seed"], "t50": r["t50"], "pred": pm, "censored": cm, "err": abs(pm - r["t50"]),
                     "err_count": abs(pi - r["t50"]),
                     "err_mean": abs(r["t50"] - float(np.mean([x["t50"] for x in tr]))), "thr": thr})
    h2 = {"runs": rows, "mae_model": float(np.mean([x["err"] for x in rows])),
          "mae_count": float(np.mean([x["err_count"] for x in rows])),
          "mae_mean": float(np.mean([x["err_mean"] for x in rows])), "n_censored": sum(x["censored"] for x in rows)}
    cnt = 0
    for _ in range(N_SWAP):
        perm = derangement(len(test), rng)
        e = [abs(predict_one(test[perm[k]]["pois"], r["lrs"], r["n_updates"], best["tau"], rows[k]["thr"])[0] - r["t50"])
             for k, r in enumerate(test)]
        cnt += float(np.mean(e)) <= h2["mae_model"]
    h2["swap_p"] = float((cnt + 1) / (N_SWAP + 1))
    h2["tier"] = tier_onset(h2["mae_model"], h2["mae_mean"], h2["mae_count"], h2["swap_p"])
    res["H2"] = h2
    # descriptive: tau the test family would pick on its own
    bt, _ = onset_fit(test, taus)
    res["test_family_own_tau"] = bt["tau"]
    return res


def with_t50(runs):
    out = []
    for r in runs:
        if r.get("status") != "ok":
            continue
        t50 = rc.first_at_least(list(r["steps"]), r["F"].mean(1), 0.5)
        if t50 is not None:
            out.append({**r, "t50": int(t50)})
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if k not in ("steps", "X", "F", "base_margin", "pois", "lrs")}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-only", action="store_true", help="report load/gate status only; compute nothing")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    fams = {"qwen_squad": load_language(a.base, SQ_ORIG, SQ_DENSE),
            "llama": load_language(a.base, LL_ORIG, LL_DENSE)}
    for fam in ("resnet50", "vit_b_16", "vim_s"):
        fams[fam] = load_classification(a.base, fam)
    res = {"loaded": {f: [{"seed": r["seed"], "status": r["status"], **({"gate_diff": r["gate_diff"]} if "gate_diff" in r else {})}
                          for r in runs] for f, runs in fams.items()}, "width": {}, "onset": {}}
    if a.load_only:
        print(json.dumps({f: [(x["seed"], x["status"], x.get("gate_diff")) for x in v] for f, v in res["loaded"].items()},
                         default=float))
        print(json.dumps({f: sum(r.get("base_margin") is not None for r in runs) for f, runs in fams.items()}))
        return
    for fam, runs in fams.items():
        recs = []
        for r in runs:
            if r.get("status") != "ok":
                continue
            w = width_record(r["steps"], r["X"], r["F"], r["base_margin"])
            w["seed"] = r["seed"]
            recs.append(w)
        summ = width_family(recs)
        res["width"][fam] = {"summary": summ, "runs": recs}
        print(json.dumps({"width": fam, **summ}, default=float), flush=True)
    sq, ll = with_t50(fams["qwen_squad"]), with_t50(fams["llama"])
    res["onset"]["squad_loo"] = loo(sq, TAUS, rng)
    print(json.dumps({"onset_squad_loo": {k: v for k, v in res["onset"]["squad_loo"].items() if k != "runs"}}, default=str), flush=True)
    res["onset"]["llama_heldout"] = held_out(sq, ll, TAUS, rng)
    h = res["onset"]["llama_heldout"]
    print(json.dumps({"onset_llama": {"tau": h["tau"], "finite_tau_wins": h["finite_tau_wins"], "H1": h["H1"],
                                      "H2": {k: v for k, v in h["H2"].items() if k != "runs"},
                                      "own_tau": h["test_family_own_tau"]}}, default=str), flush=True)
    with open(a.out, "w") as fh:
        json.dump(clean(res), fh, indent=1, default=float)


if __name__ == "__main__":
    main()
