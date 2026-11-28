"""D-check v3: family-population analysis (CPU). Design: ../ROOT_CAUSE_v3.zh-CN.md; contract: ../CONTRACT_DCHECK_v3.zh-CN.md.

Unit of inference = attack family. For a held-out family h and a canary set S of k generator families:
  the defender inserts K_PER canaries per family in S, the dataset holds M_REAL rows of h; the filter flags at the
  (1 - FPR) quantile of the evaluation task's clean rows; the audit takes r_L = min over f in S of the one-sided
  Clopper-Pearson lower bound of family f's canary recall (Bonferroni alpha/k), then the binomial inversion of WST5,
  U = max P - d s.t. P(Bin(P, r_L) <= d) >= ALPHA; the audit FAILS when the true residual M_REAL - d exceeds U.
  F_h(k) = mean over N_SETS random canary sets of P(audit fails)   (inner Monte Carlo, R_INNER draws)
  rank_h(k) = mean over the same sets of 1[r_h < min_{f in S} r_f]   (exchangeable reference: 1/(k+1), less with ties)
Population statistic: Delta(k) = mean_{external} F_h - mean_{generator, leave-one-out} F_g.
  p   : one-sided permutation test over family labels (approximate: canaries always come from the generator pool)
  CI  : 90% bootstrap over families (external and generator resampled separately)
Verdict (main: MAIN_FILTER, FPR_MAIN, K_MAIN):
  NON_EXCHANGEABLE  iff p < 0.05 and Delta >= EXCESS_TOL
  CONSISTENT        iff CI upper < EXCESS_TOL
  INDETERMINATE     otherwise
Null gate G3: pseudo-external = N_PSEUDO random generator families (canary pool = the remaining ones), repeated
  N_G3 times; fraction NON_EXCHANGEABLE must be <= G3_MAX, else CRITERION_BROKEN.
usage: python analyze_v3.py --scores scores.jsonl --external-class receipts/external_class.json --out v3.json
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
N_SETS = 100
R_INNER = 2000
EXCESS_TOL = 0.05
N_PERM = 10000
N_BOOT = 2000
N_PSEUDO = 10
N_G3 = 200
N_SETS_G3 = 20
R_INNER_G3 = 500
G3_MAX = 0.08
GEN_TASK = "alpaca"
SEED = 20261129


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


def family_stats(r_h, pool_recalls, k, rng, n_sets=N_SETS, R=R_INNER):
    """pool_recalls: recalls of the candidate canary families (h excluded)."""
    pool = np.asarray(pool_recalls)
    F = rk = 0.0
    for _ in range(n_sets):
        S = pool[rng.choice(pool.size, size=k, replace=False)]
        F += fail_prob(r_h, S, rng, R=R)
        rk += float(r_h < S.min())
    return F / n_sets, rk / n_sets


def population_verdict(F_ext, F_gen, rng, n_perm=N_PERM, n_boot=N_BOOT):
    F_ext, F_gen = np.asarray(F_ext), np.asarray(F_gen)
    delta = float(F_ext.mean() - F_gen.mean())
    allF = np.concatenate([F_ext, F_gen])
    ne = F_ext.size
    perm = np.array([np.mean(x[:ne]) - np.mean(x[ne:]) for x in (rng.permutation(allF) for _ in range(n_perm))])
    p = float((np.sum(perm >= delta) + 1) / (n_perm + 1))
    boots = np.array([F_ext[rng.integers(0, ne, ne)].mean() - F_gen[rng.integers(0, F_gen.size, F_gen.size)].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [5, 95])
    if p < 0.05 and delta >= EXCESS_TOL:
        v = "NON_EXCHANGEABLE"
    elif hi < EXCESS_TOL:
        v = "CONSISTENT"
    else:
        v = "INDETERMINATE"
    return {"delta": delta, "p_perm": p, "ci90": [float(lo), float(hi)], "verdict": v}


def recalls(by, task, filt, fams, fpr):
    clean, sc = ctx_scores(by, task, filt, fams)
    t = thr(clean, fpr)
    return {f: float(np.mean(sc[f] > t)) for f in fams}


def run(rows, external_class, filt=MAIN_FILTER, fpr=FPR_MAIN, ks=K_LIST, seed=SEED, g3=True,
        n_sets=N_SETS, R=R_INNER, n_g3=N_G3):
    """external_class: {family: {"task": eval task, "class": "seen"|"unseen", "source": str}}."""
    by = index_rows(rows)
    rng = np.random.default_rng(seed)
    gen = sorted(f for f in by if f.startswith("gen:"))
    out = {"filter": filt, "fpr": fpr, "n_generator_families": len(gen), "by_k": {}}
    rg = recalls(by, GEN_TASK, filt, gen, fpr)
    out["generator_recalls"] = rg
    ext_rec = {}
    for h, meta in external_class.items():
        rr = recalls(by, meta["task"], filt, gen + [h], fpr)
        ext_rec[h] = rr
    out["external_recalls"] = {h: v[h] for h, v in ext_rec.items()}
    for k in ks:
        Fg, Rg = {}, {}
        for g in gen:
            pool = [rg[x] for x in gen if x != g]
            Fg[g], Rg[g] = family_stats(rg[g], pool, k, rng, n_sets, R)
        Fe, Re = {}, {}
        for h, rr in ext_rec.items():
            Fe[h], Re[h] = family_stats(rr[h], [rr[x] for x in gen], k, rng, n_sets, R)
        res = {"F_generator": Fg, "rank_generator": Rg, "F_external": Fe, "rank_external": Re,
               "rank_reference": 1 / (k + 1), "mean_rank_generator": float(np.mean(list(Rg.values())))}
        for cls in ("unseen", "seen"):
            hs = [h for h, m in external_class.items() if m["class"] == cls]
            if hs:
                res[f"verdict_{cls}"] = population_verdict([Fe[h] for h in hs], list(Fg.values()), rng)
        out["by_k"][str(k)] = res
    if g3:
        out["g3"] = g3_null(rg, gen, rng, n_g3)
    main = out["by_k"][str(K_MAIN)]
    broken = g3 and out["g3"]["non_exchangeable_rate"] > G3_MAX
    out["main_verdict"] = ("CRITERION_BROKEN" if broken else
                           main.get("verdict_unseen", {"verdict": "NO_UNSEEN_FAMILIES"})["verdict"])
    out["seen_class_verdict"] = main.get("verdict_seen", {}).get("verdict")
    return out


def g3_null(rg, gen, rng, n_g3=N_G3, k=K_MAIN):
    hits = 0
    for _ in range(n_g3):
        ps = set(rng.choice(len(gen), size=N_PSEUDO, replace=False).tolist())
        pseudo = [gen[i] for i in ps]
        rest = [gen[i] for i in range(len(gen)) if i not in ps]
        Fe = [family_stats(rg[h], [rg[x] for x in rest], k, rng, N_SETS_G3, R_INNER_G3)[0] for h in pseudo]
        Fg = [family_stats(rg[g], [rg[x] for x in rest if x != g], k, rng, N_SETS_G3, R_INNER_G3)[0] for g in rest]
        hits += population_verdict(Fe, Fg, rng, n_perm=2000, n_boot=500)["verdict"] == "NON_EXCHANGEABLE"
    return {"non_exchangeable_rate": hits / n_g3, "n": n_g3}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--external-class", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.scores) if l.strip()]
    ext = json.load(open(a.external_class))
    res = {"main": run(rows, ext)}
    res["secondary"] = {f"{filt}@{fpr}": run(rows, ext, filt=filt, fpr=fpr, ks=(K_MAIN,), g3=False)["by_k"]
                        for filt in FILTERS for fpr in FPR_LIST if not (filt == MAIN_FILTER and fpr == FPR_MAIN)}
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    m = res["main"]
    print(json.dumps({"main_verdict": m["main_verdict"], "seen_class_verdict": m["seen_class_verdict"],
                      "g3": m.get("g3")}, indent=1))
