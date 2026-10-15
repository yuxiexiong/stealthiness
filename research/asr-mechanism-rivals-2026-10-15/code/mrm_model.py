"""Minimal readout model (CONTRACT_MRM.md). numpy only, read-only.

Two independent halves, each falsifiable.

WIDTH  ASR(t) = fraction of probes with c(t) + theta_i > 0.
       theta_i (per-probe offset) is read two ways: (BASE) from the step-0 base model margins, (WINDOW) from
       the atlas additive fit inside the window. c(t) is the per-probe mean margin trajectory. We ask how well
       the predicted window width W_hat = theta-band / in-window speed matches the observed width, and whether
       BASE theta predicts it as well as WINDOW theta. If BASE works, the band is set before training.

ONSET  A leaky accumulator x(t) = sum over poison updates u<=t of lr(u) * exp(-(t-u)/tau). Fit one shared tau on
       the fit seeds so that x at the per-seed onset t50 has minimum coefficient of variation; compare to tau=inf
       (plain LR-weighted count). Then hold tau fixed and test on held-out seeds: predict each held-out onset as
       the step where x first reaches the fit-seeds' median threshold; report the error. Falsifiable: if no tau
       beats tau=inf, or held-out error is no better than predicting the mean onset, there is no such accumulator.

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

SQUAD = {s: (f"20261004_lora/runs/llm_s{s}_poison" if s <= 1003 else f"20261004_seed10/runs/llm_s{s}_poison")
         for s in range(1001, 1011)}
LLAMA = {s: f"20261007_vim_llama_v1/llama/runs/formal_s{s}_poison" for s in range(1001, 1006)}
TAUS = [5, 10, 20, 40, 80, 160, 320, 640, math.inf]


# ---------------- WIDTH ----------------
def width_record(steps, X, F, base_margin):
    """base_margin: dict id->step0 trigger margin, aligned to this run's probe order upstream."""
    steps = list(steps)
    asr = F.mean(1)
    w = rc.window(steps, asr)
    if not w:
        return None
    a, b = w
    ar = rc.atlas_ratio(steps, X.mean(1), a, b)
    if not ar:
        return None
    p, ia, ib = ar["p"], steps.index(a), steps.index(b)
    ip = steps.index(p)
    Xs = X[ip:ib + 1]
    theta_win = Xs.mean(0) - Xs.mean()                  # window additive offset
    theta_base = base_margin - base_margin.mean()       # base-model offset, same probes
    v_in = ar["v_in"]
    if not (v_in > 0):
        return None
    out = {"window": [p, a, b], "v_in": v_in, "obs_width": b - a}
    for name, th in (("win", theta_win), ("base", theta_base)):
        band = float(np.quantile(th, 0.95) - np.quantile(th, 0.05))
        out[name] = {"band": band, "pred_width": band / v_in,
                     "log2_pred_over_obs": float(np.log2((band / v_in) / (b - a))) if (b - a) > 0 else None}
    return out


# ---------------- ONSET ----------------
def accumulator(pois, lrs, tau, upto):
    """x(t) for t in 1..upto: leaky sum of lr(u) over poison-carrying updates u<=t."""
    x = np.zeros(upto + 1)
    contrib = {u: lrs.get(u, 0.0) * pois.get(u, 0) for u in range(1, upto + 1)}
    for t in range(1, upto + 1):
        if tau == math.inf:
            x[t] = x[t - 1] + contrib[t]
        else:
            x[t] = x[t - 1] * math.exp(-1.0 / tau) + contrib[t]
    return x


def onset_fit(runs, taus):
    """For each tau: x at each run's t50; report coefficient of variation across runs."""
    best = None
    table = []
    for tau in taus:
        vals = []
        for r in runs:
            x = accumulator(r["pois"], r["lrs"], tau, r["t50"])
            vals.append(x[r["t50"]])
        vals = np.array(vals)
        cv = float(vals.std() / vals.mean()) if vals.mean() > 0 else math.inf
        table.append({"tau": tau if tau != math.inf else "inf", "cv": cv, "median_threshold": float(np.median(vals))})
        if best is None or cv < best["cv"]:
            best = {"tau": tau, "cv": cv, "median_threshold": float(np.median(vals))}
    return best, table


def onset_predict(runs, tau, thr):
    """Predict each run's onset as the first step where x(t) >= thr; error vs observed t50."""
    out = []
    for r in runs:
        x = accumulator(r["pois"], r["lrs"], tau, r["t50"] + 200)
        pred = next((t for t in range(1, len(x)) if x[t] >= thr), None)
        out.append({"seed": r["seed"], "t50": r["t50"], "pred": pred,
                    "err": abs(pred - r["t50"]) if pred is not None else None})
    return out


def load_family(base, mapping):
    runs = []
    base_margins = {}
    for s, rel in mapping.items():
        d = os.path.join(base, rel)
        got = rl.language_run(d)
        if got is None:
            continue
        steps, X, F, ids, Xc, acc = got
        asr = F.mean(1)
        t50 = rc.first_at_least(list(steps), asr, 0.5)
        if t50 is None:
            continue
        s0 = {r["id"]: r["trigger"]["margin"] for r in (json.loads(l) for l in open(os.path.join(d, "samples-0000.jsonl")))}
        bm = np.array([s0[i] for i in ids], float)
        tr = [json.loads(l) for l in open(os.path.join(d, "training.jsonl"))]
        pois = {r["optimizer_step"]: r["poison_examples"] for r in tr}
        lrs = {r["optimizer_step"]: r["lr"] for r in tr}
        runs.append({"seed": s, "steps": steps, "X": X, "F": F, "ids": ids, "base_margin": bm,
                     "t50": t50, "pois": pois, "lrs": lrs})
        base_margins[s] = bm
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sq = load_family(a.base, SQUAD)
    ll = load_family(a.base, LLAMA)
    res = {"width": {}, "onset": {}}

    # WIDTH, both families
    for name, runs in (("qwen_squad", sq), ("llama", ll)):
        recs = [width_record(r["steps"], r["X"], r["F"], r["base_margin"]) for r in runs]
        recs = [x for x in recs if x]
        res["width"][name] = {
            "n": len(recs),
            "median_abs_log2_win": rc.med([abs(x["win"]["log2_pred_over_obs"]) for x in recs]),
            "median_abs_log2_base": rc.med([abs(x["base"]["log2_pred_over_obs"]) for x in recs]),
            "median_band_win": rc.med([x["win"]["band"] for x in recs]),
            "median_band_base": rc.med([x["base"]["band"] for x in recs]),
            "runs": [{k: x[k] for k in ("window", "v_in", "obs_width", "win", "base")} for x in recs]}
        print(json.dumps({("width/" + name): {k: res["width"][name][k] for k in
              ("n", "median_abs_log2_win", "median_abs_log2_base")}}, default=float), flush=True)

    # ONSET primary: leave-one-out within SQuAD (fit tau + threshold on 9 seeds, predict the 10th)
    loo = []
    for k in range(len(sq)):
        train = sq[:k] + sq[k + 1:]
        bk, _ = onset_fit(train, TAUS)
        pk = onset_predict([sq[k]], bk["tau"], bk["median_threshold"])[0]
        loo.append({"seed": sq[k]["seed"], "tau": bk["tau"] if bk["tau"] != math.inf else "inf",
                    "t50": pk["t50"], "pred": pk["pred"], "err": pk["err"]})
    loo_err = [x["err"] for x in loo if x["err"] is not None]
    loo_base = [abs(sq[k]["t50"] - float(np.mean([r["t50"] for j, r in enumerate(sq) if j != k]))) for k in range(len(sq))]
    # count-only LOO (tau = inf fixed), the known-weak null
    loo_inf = []
    for k in range(len(sq)):
        train = sq[:k] + sq[k + 1:]
        _, tb = onset_fit(train, [math.inf])
        pk = onset_predict([sq[k]], math.inf, tb[0]["median_threshold"])[0]
        loo_inf.append(pk["err"])
    res["onset_loo"] = {"runs": loo, "mean_abs_err": float(np.mean(loo_err)) if loo_err else None,
                        "mean_abs_err_count_only": float(np.mean([e for e in loo_inf if e is not None])) if any(e is not None for e in loo_inf) else None,
                        "mean_abs_err_mean_baseline": float(np.mean(loo_base)),
                        "n_unpredicted": sum(x["err"] is None for x in loo)}
    print(json.dumps({"onset_loo": {k: res["onset_loo"][k] for k in ("mean_abs_err", "mean_abs_err_count_only",
          "mean_abs_err_mean_baseline", "n_unpredicted")}}, default=float), flush=True)

    # ONSET secondary: fit tau on all SQuAD, test on Llama
    best, table = onset_fit(sq, TAUS)
    inf_cv = next(t["cv"] for t in table if t["tau"] == "inf")
    fit_pred = onset_predict(sq, best["tau"], best["median_threshold"])
    test_pred = onset_predict(ll, best["tau"], best["median_threshold"])
    mean_onset_sq = float(np.mean([r["t50"] for r in sq]))
    baseline_err = float(np.mean([abs(r["t50"] - mean_onset_sq) for r in ll]))
    test_err = [p["err"] for p in test_pred if p["err"] is not None]
    res["onset"] = {"taus": table, "best_tau": best["tau"] if best["tau"] != math.inf else "inf", "best_cv": best["cv"],
                    "inf_cv": inf_cv, "fit_threshold": best["median_threshold"], "fit_pred": fit_pred,
                    "test_pred": test_pred, "test_mean_abs_err": float(np.mean(test_err)) if test_err else None,
                    "baseline_mean_abs_err": baseline_err,
                    "beats_count": bool(best["cv"] < 0.8 * inf_cv),
                    "beats_mean_baseline": bool(test_err and np.mean(test_err) < baseline_err)}
    print(json.dumps({"onset": {k: res["onset"][k] for k in
          ("best_tau", "best_cv", "inf_cv", "test_mean_abs_err", "baseline_mean_abs_err",
           "beats_count", "beats_mean_baseline")}}, default=float), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
