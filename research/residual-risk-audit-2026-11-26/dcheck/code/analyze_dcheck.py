"""D-check verdicts from scores.jsonl (CPU). Contract: ../CONTRACT_DCHECK.zh-CN.md (all constants below are frozen there).

Flag rule: threshold = (1 - FPR) quantile of the CLEAN rows' scores, per filter; a row is flagged iff score > threshold.
Recall of a family = fraction of its rows flagged. NaN scores are never flagged (counted and reported).

D test (per eligible filter, per held-out family h, canary set C):
  Delta = recall(h) - min_{c in C} recall(c); 90% percentile bootstrap CI (TOST: each one-sided claim at alpha 0.05); token families are resampled jointly by
  base instruction (paired design), the topic family and the clean rows independently; the threshold is re-estimated
  from the resampled clean rows in every bootstrap draw.
  HOLDS  iff CI_lo >= -DELTA;  VIOLATED iff CI_hi < -DELTA;  INDETERMINATE otherwise.
Eligibility: pooled poison recall (all four families) one-sided 95% Clopper-Pearson lower bound > FPR + 0.10.
Null check: split each token family's bases at random into halves A/B (paired across nothing; one family at a time);
  run the same D test with held-out = A, canaries = {B}; fraction VIOLATED over all splits must be <= NULL_MAX_VIOL.
usage: python analyze_dcheck.py --scores scores.jsonl --out verdicts.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

BASE_FILTERS = ("resp_loss", "onion", "gs_entropy")
# Union of two filters at one total FPR: each score is mapped to its clean-row empirical CDF, the union score is the max.
UNIONS = (("resp_loss", "onion"), ("resp_loss", "gs_entropy"), ("onion", "gs_entropy"))
FILTERS = BASE_FILTERS + tuple(f"u:{a}+{b}" for a, b in UNIONS)
TOKEN_FAMS = ("badnet", "vpi", "mtba")
ALL_FAMS = TOKEN_FAMS + ("topic",)
FPR_LIST = (0.05, 0.10)
FPR_MAIN = 0.05
DELTA = 0.05
B_MAIN = 10000
B_NULL = 2000
N_SPLITS = 200
NULL_MAX_VIOL = 0.10
ELIG_MARGIN = 0.10
GS_DEFAULT_THRESHOLD = 0.822  # GradSentry genConfigs/svd.json (descriptive only)
SEED = 20261128
CI_PCT = (5.0, 95.0)


def cp_lower(x, n, a=0.05):
    return 0.0 if x == 0 else float(stats.beta.ppf(a, x, n - x + 1))


def prepare(rows, filt):
    s = lambda r: (r[filt] if r[filt] is not None and not np.isnan(r[filt]) else -np.inf)
    clean = np.array([s(r) for r in rows if r["family"] == "clean"])
    fam = {}
    for f in TOKEN_FAMS:
        d = {r["base_idx"]: s(r) for r in rows if r["family"] == f}
        fam[f] = d
    bases = sorted(set.intersection(*[set(fam[f]) for f in TOKEN_FAMS]))
    tok = np.array([[fam[f][b] for b in bases] for f in TOKEN_FAMS])  # (3, nb), paired by base
    topic = np.array([s(r) for r in rows if r["family"] == "topic"])
    n_nan = int(sum(1 for r in rows if r[filt] is None or np.isnan(r[filt])))
    return clean, tok, topic, n_nan


def thr(clean, fpr):
    return np.quantile(clean, 1 - fpr, method="higher")


def recalls(clean, tok, topic, fpr):
    t = thr(clean, fpr)
    r = {f: float(np.mean(tok[i] > t)) for i, f in enumerate(TOKEN_FAMS)}
    r["topic"] = float(np.mean(topic > t)) if topic.size else float("nan")
    r["clean_flag_rate"] = float(np.mean(clean > t))
    return r, t


def boot_delta(clean, tok, topic, fpr, held, canaries, B, rng):
    """Bootstrap distribution of recall(held) - min recall(canaries)."""
    nc, nb, nt = clean.size, tok.shape[1], topic.size
    out = np.empty(B)
    idx_f = {f: i for i, f in enumerate(TOKEN_FAMS)}
    for b in range(B):
        t = thr(clean[rng.integers(0, nc, nc)], fpr)
        bi = rng.integers(0, nb, nb)
        rec = {f: np.mean(tok[idx_f[f], bi] > t) for f in TOKEN_FAMS}
        if nt:
            rec["topic"] = np.mean(topic[rng.integers(0, nt, nt)] > t)
        out[b] = rec[held] - min(rec[c] for c in canaries)
    return out


def verdict(lo, hi, delta=DELTA):
    if lo >= -delta:
        return "HOLDS"
    if hi < -delta:
        return "VIOLATED"
    return "INDETERMINATE"


def eligible(clean, tok, topic, fpr):
    t = thr(clean, fpr)
    allp = np.concatenate([tok.reshape(-1), topic])
    x = int(np.sum(allp > t))
    lo = cp_lower(x, allp.size)
    return lo > fpr + ELIG_MARGIN, {"pooled_recall": x / allp.size, "cp_lower": lo}


def d_tests(clean, tok, topic, fpr, B, rng):
    res = {}
    plan = [(h, tuple(c for c in TOKEN_FAMS if c != h)) for h in TOKEN_FAMS]
    if topic.size:
        plan.append(("topic", TOKEN_FAMS))
    rec, _ = recalls(clean, tok, topic, fpr)
    for h, C in plan:
        obs = rec[h] - min(rec[c] for c in C)
        bd = boot_delta(clean, tok, topic, fpr, h, C, B, rng)
        lo, hi = np.percentile(bd, CI_PCT)
        res[h] = {"canaries": list(C), "delta": obs, "ci": [float(lo), float(hi)], "verdict": verdict(lo, hi)}
    return res


def null_check(clean, tok, fpr, rng, n_splits=N_SPLITS, B=B_NULL):
    """Split one token family's bases in halves; D test A vs {B}. Returns fraction VIOLATED."""
    nb = tok.shape[1]
    viol = tot = 0
    for i in range(len(TOKEN_FAMS)):
        for _ in range(n_splits):
            perm = rng.permutation(nb)
            A, Bh = tok[i, perm[: nb // 2]], tok[i, perm[nb // 2:]]
            out = np.empty(B)
            for b in range(B):
                t = thr(clean[rng.integers(0, clean.size, clean.size)], fpr)
                out[b] = np.mean(A[rng.integers(0, A.size, A.size)] > t) - np.mean(Bh[rng.integers(0, Bh.size, Bh.size)] > t)
            lo, hi = np.percentile(out, CI_PCT)
            viol += verdict(lo, hi) == "VIOLATED"
            tot += 1
    return viol / tot


def filter_verdict(entry):
    """Per filter: SUPPORTED iff every held-out family HOLDS; VIOLATED iff any is VIOLATED; else INDETERMINATE."""
    if not entry["eligible"]:
        return "NOT_ELIGIBLE"
    vs = [v["verdict"] for v in entry["d"].values()]
    if "VIOLATED" in vs:
        return "VIOLATED"
    return "SUPPORTED" if all(v == "HOLDS" for v in vs) else "INDETERMINATE"


def overall(per_filter):
    fv = {f: filter_verdict(e) for f, e in per_filter.items()}
    for f, e in per_filter.items():
        e["filter_verdict"] = fv[f]
    if all(v == "NOT_ELIGIBLE" for v in fv.values()):
        return "NO_ELIGIBLE_FILTER"
    if any(v == "VIOLATED" for v in fv.values()):
        return "D_VIOLATED"
    if any(v == "SUPPORTED" for v in fv.values()):
        return "D_SUPPORTED"
    return "D_INDETERMINATE"


def odds_ratio(f1, f2):
    a = np.sum(f1 & f2); b = np.sum(f1 & ~f2); c = np.sum(~f1 & f2); d = np.sum(~f1 & ~f2)
    return float((a + .5) * (d + .5) / ((b + .5) * (c + .5)))


def add_union_scores(rows):
    """Clean-ECDF transform of each base filter, then max over the pair. NaN stays NaN only if both are NaN."""
    rows = [dict(r) for r in rows]
    ecdf = {}
    for f in BASE_FILTERS:
        c = np.sort([r[f] for r in rows if r["family"] == "clean" and r[f] is not None and not np.isnan(r[f])])
        ecdf[f] = c
    def F(f, x):
        if x is None or np.isnan(x):
            return np.nan
        return np.searchsorted(ecdf[f], x, side="right") / ecdf[f].size
    for r in rows:
        for a, b in UNIONS:
            va, vb = F(a, r[a]), F(b, r[b])
            r[f"u:{a}+{b}"] = float(np.nanmax([va, vb])) if not (np.isnan(va) and np.isnan(vb)) else float("nan")
    return rows


def analyze(rows, B=B_MAIN, n_splits=N_SPLITS, b_null=B_NULL, seed=SEED, with_null=True, with_or=True):
    rows = add_union_scores(rows)
    rng = np.random.default_rng(seed)
    out = {"constants": {"FPR_MAIN": FPR_MAIN, "DELTA": DELTA, "B": B, "N_SPLITS": n_splits, "B_NULL": b_null,
                         "ELIG_MARGIN": ELIG_MARGIN, "NULL_MAX_VIOL": NULL_MAX_VIOL}, "by_fpr": {}}
    for fpr in FPR_LIST:
        pf = {}
        for filt in FILTERS:
            clean, tok, topic, n_nan = prepare(rows, filt)
            ok, el = eligible(clean, tok, topic, fpr)
            rec, t = recalls(clean, tok, topic, fpr)
            entry = {"eligible": bool(ok), "eligibility": el, "recall": rec, "threshold": float(t), "n_nan": n_nan}
            entry["d"] = d_tests(clean, tok, topic, fpr, B, rng) if ok else {}
            if with_null and ok and fpr == FPR_MAIN:
                entry["null_violation_rate"] = null_check(clean, tok, fpr, rng, n_splits, b_null)
            pf[filt] = entry
        out["by_fpr"][str(fpr)] = {"filters": pf, "overall": overall(pf)}
    main = out["by_fpr"][str(FPR_MAIN)]
    broken = [f for f, e in main["filters"].items() if e.get("null_violation_rate", 0) > NULL_MAX_VIOL]
    out["criterion_broken_filters"] = broken
    out["main_verdict"] = "CRITERION_BROKEN" if broken else main["overall"]
    if with_or:  # D' descriptive: OR between filter pairs, held-out vs pooled canaries, at FPR_MAIN
        flags = {}
        for filt in BASE_FILTERS:
            clean, tok, topic, _ = prepare(rows, filt)
            t = thr(clean, FPR_MAIN)
            flags[filt] = {"tok": tok > t, "topic": topic > t}
        ors = {}
        for i, f1 in enumerate(BASE_FILTERS):
            for f2 in BASE_FILTERS[i + 1:]:
                d = {}
                for k, h in enumerate(TOKEN_FAMS):
                    others = [j for j in range(3) if j != k]
                    d[h] = {"held_out_or": odds_ratio(flags[f1]["tok"][k], flags[f2]["tok"][k]),
                            "canary_or": odds_ratio(flags[f1]["tok"][others].reshape(-1),
                                                    flags[f2]["tok"][others].reshape(-1))}
                if flags[f1]["topic"].size:
                    d["topic"] = {"held_out_or": odds_ratio(flags[f1]["topic"], flags[f2]["topic"]),
                                  "canary_or": odds_ratio(flags[f1]["tok"].reshape(-1), flags[f2]["tok"].reshape(-1))}
                ors[f"{f1}|{f2}"] = d
        out["d_prime_descriptive"] = ors
    gsr = [r for r in rows if r["family"] != "clean"]
    out["gs_default_threshold_recall"] = {f: float(np.mean([r["gs_entropy"] > GS_DEFAULT_THRESHOLD
                                                             for r in gsr if r["family"] == f])) for f in ALL_FAMS
                                          if any(r["family"] == f for r in gsr)}
    out["gs_default_threshold_clean_flag_rate"] = float(np.mean([r["gs_entropy"] > GS_DEFAULT_THRESHOLD
                                                                 for r in rows if r["family"] == "clean"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.scores) if l.strip()]
    res = analyze(rows)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"main_verdict": res["main_verdict"],
                      "filters": {f: {"filter_verdict": e["filter_verdict"], "recall": e["recall"],
                                      "d": {h: (v["verdict"], round(v["delta"], 3)) for h, v in e["d"].items()},
                                      "null_viol": e.get("null_violation_rate")}
                                  for f, e in res["by_fpr"][str(FPR_MAIN)]["filters"].items()}}, indent=1))
