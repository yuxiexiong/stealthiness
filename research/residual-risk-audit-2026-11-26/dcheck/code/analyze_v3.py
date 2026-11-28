"""D-check v3: per-family conformal test against the frozen canary-family generator (CPU).
Design: ../ROOT_CAUSE_v3.zh-CN.md; contract: ../CONTRACT_DCHECK_v3.zh-CN.md.

Audit model (consumer = the WST5 audit). For a held-out family h evaluated in data context T and a canary set S of
k generator families applied to T's own clean pool: K_PER canaries per family, M_REAL rows of h; flags above the
(1 - FPR) quantile of T's clean rows; r_L = min over f in S of the one-sided Clopper-Pearson lower bound (alpha/k);
U = max P - d with P(Bin(P, r_L) <= d) >= ALPHA (binomial inversion); the audit FAILS when M_REAL - d > U.
  F_h = mean over N_SETS random canary sets of P(audit fails)            (inner Monte Carlo, R_INNER draws)
Conformal test (unit = family). In context T, every generator family g gets F_g (leave-one-out: its canary sets come
from the other generator families). Under exchangeability of h with the generator families,
  p_h = (1 + #{g : F_g >= F_h}) / (n_gen + 1)
is a valid p-value. OUTSIDE iff p_h <= ALPHA_FAM for every family (no Bonferroni: with ~40 generator families the
smallest attainable p is 1/(n_gen+1) ~ 0.025, so Bonferroni over two main anchors would make OUTSIDE unreachable);
the family-wise error over the main anchors is reported as 1 - (1 - ALPHA_FAM)^n_main.
Main verdict: CRITERION_BROKEN (gate G3) | D_VIOLATED (some main anchor OUTSIDE) | D_NOT_REJECTED.
The test never claims support for D; D_NOT_REJECTED is reported with the rehearsed power.
Gate G3 (implementation check): within each context, the generator families' own leave-one-out p-values must not
fall at or below ALPHA_FAM more often than G3_MAX.
usage: python analyze_v3.py --scores scores.jsonl --external-class external_class.json --out v3.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from analyze_dcheck import FILTERS, UNION3, ctx_scores, index_rows, thr

MAIN_FILTER = UNION3
FPR_LIST = (0.01, 0.05, 0.10)
FPR_MAIN = 0.05
K_LIST = (3, 5, 10)
K_MAIN = 5
K_PER = 100
M_REAL = 200
ALPHA = 0.05
ALPHA_FAM = 0.05
N_SETS = 100
R_INNER = 2000
G3_MAX = 0.10
SEED = 20261130


def fail_prob(r_h, r_s, rng, m=M_REAL, K=K_PER, R=R_INNER, alpha=ALPHA):
    k = len(r_s)
    kf = rng.binomial(K, np.asarray(r_s, float)[:, None], size=(k, R))
    a = alpha / k
    L = np.where(kf > 0, stats.beta.ppf(a, np.maximum(kf, 1), K - kf + 1), 0.0).min(axis=0)
    d = rng.binomial(m, r_h, size=R)
    U = np.full(R, np.inf)
    ok = L > 0
    if ok.any():
        x = stats.nbinom.ppf(1 - alpha, d[ok] + 1, L[ok])
        eq = stats.nbinom.cdf(x, d[ok] + 1, L[ok]) <= 1 - alpha + 1e-12
        U[ok] = x + eq
    return float(np.mean((m - d) > U))


def F_of(r_h, pool, k, rng, n_sets=N_SETS, R=R_INNER):
    pool = np.asarray(pool)
    return float(np.mean([fail_prob(r_h, pool[rng.choice(pool.size, size=k, replace=False)], rng, R=R)
                          for _ in range(n_sets)]))


def conformal_p(F_h, F_gen):
    F_gen = np.asarray(F_gen)
    return float((1 + np.sum(F_gen >= F_h)) / (F_gen.size + 1))


def context_null(rec_gen, k, rng, n_sets, R):
    gen = sorted(rec_gen)
    F = {g: F_of(rec_gen[g], [rec_gen[x] for x in gen if x != g], k, rng, n_sets, R) for g in gen}
    p_loo = {g: float((1 + sum(F[x] >= F[g] for x in gen if x != g)) / len(gen)) for g in gen}
    return F, p_loo


def run(rows, ext, filt=MAIN_FILTER, fpr=FPR_MAIN, k=K_MAIN, seed=SEED, n_sets=N_SETS, R=R_INNER):
    by = index_rows(rows)
    rng = np.random.default_rng(seed)
    ctxs = sorted({(m["task"], m["gen_prefix"]) for m in ext.values()})
    n_main = sum(1 for m in ext.values() if m["role"] == "main")
    out = {"filter": filt, "fpr": fpr, "k": k, "contexts": {}, "families": {}}
    for T, P in ctxs:
        gen = sorted(f for f in by if f.startswith(P + ":"))
        hs = [h for h, m in ext.items() if (m["task"], m["gen_prefix"]) == (T, P)]
        clean, sc = ctx_scores(by, T, filt, gen + hs)
        t = thr(clean, fpr)
        rec = {f: float(np.mean(sc[f] > t)) for f in gen + hs}
        rg = {g: rec[g] for g in gen}
        Fg, p_loo = context_null(rg, k, rng, n_sets, R)
        g3 = float(np.mean([p <= ALPHA_FAM for p in p_loo.values()]))
        Fv = np.array(list(Fg.values()))
        out["contexts"][f"{T}|{P}"] = {"n_gen": len(gen), "g3_rate": g3, "F_gen_quantiles":
                                       {q: float(np.quantile(Fv, q)) for q in (0.5, 0.9, 1.0)},
                                       "gen_recall_quantiles": {q: float(np.quantile(list(rg.values()), q))
                                                                for q in (0.0, 0.1, 0.5, 0.9)},
                                       "clean_flag_rate": float(np.mean(clean > t))}
        for h in hs:
            F_h = F_of(rec[h], list(rg.values()), k, rng, n_sets, R)
            p = conformal_p(F_h, Fv)
            role = ext[h]["role"]
            a = ALPHA_FAM
            out["families"][h] = {**ext[h], "recall": rec[h], "F": F_h, "p": p, "alpha_used": a,
                                  "outside": bool(p <= a), "context": f"{T}|{P}"}
    broken = [c for c, v in out["contexts"].items() if v["g3_rate"] > G3_MAX]
    viol = [h for h, v in out["families"].items() if v["role"] == "main" and v["outside"]]
    out["criterion_broken_contexts"] = broken
    out["main_verdict"] = "CRITERION_BROKEN" if broken else ("D_VIOLATED" if viol else "D_NOT_REJECTED")
    out["violating_main_families"] = viol
    out["main_fwer_bound"] = 1 - (1 - ALPHA_FAM) ** max(1, n_main)
    out["min_attainable_p"] = {c: 1 / (v["n_gen"] + 1) for c, v in out["contexts"].items()}
    for role in ("main", "secondary", "descriptive"):
        fs = [v for v in out["families"].values() if v["role"] == role]
        if fs:
            out[f"summary_{role}"] = {"n": len(fs), "n_outside": sum(v["outside"] for v in fs),
                                      "median_p": float(np.median([v["p"] for v in fs]))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--external-class", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.scores) if l.strip()]
    ext = json.load(open(a.external_class))
    res = {"main": run(rows, ext)}
    res["sensitivity"] = {f"{f}@{fpr}@k{k}": run(rows, ext, filt=f, fpr=fpr, k=k, n_sets=30, R=500)
                          for f in FILTERS for fpr in FPR_LIST for k in K_LIST
                          if not (f == MAIN_FILTER and fpr == FPR_MAIN and k == K_MAIN)
                          and (f == MAIN_FILTER or (fpr == FPR_MAIN and k == K_MAIN))}
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    m = res["main"]
    print(json.dumps({"main_verdict": m["main_verdict"], "violating": m["violating_main_families"],
                      "contexts": {c: v["g3_rate"] for c, v in m["contexts"].items()},
                      "families": {h: (round(v["recall"], 3), round(v["F"], 3), round(v["p"], 3), v["role"])
                                   for h, v in m["families"].items()}}, indent=1))
