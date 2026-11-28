"""L2 pilot: filter-selected subsets and the zero-GPU gates (CPU). Run right after l2_score.py; commit the scores
file and this output's sha immediately (review round 3 S3). make_l2_jobs re-derives the subsets with select().

Clean reference: for resp_loss and gs_literal, every clean row (19,800); for onion and u3 (which needs all three
scores), the design's ONION clean sample (4000 rows) (review round 3: FPR* standard error at the 0.005 / 0.10 edges).
For each filter F (higher = more suspicious) and each k (the dose grid, plus kstar if given):
  leaked(F, k) = the k lowest-scoring poison rows (ties broken by sha256(id)); t = the k-th of these scores;
  implied FPR*(F, k) = share of the clean reference with score > t; caught(F, k) = the k highest-scoring poison rows.
Load-bearing(F, k) iff WP_FPR[0] <= FPR* <= WP_FPR[1], F has >= MIN_DISTINCT distinct poison scores, and no other
poison row ties with t (otherwise "leaked = every row the threshold lets through" would not hold).
usage: python l2_select.py --scores l2_scores.jsonl [--kstar K] --out l2_subsets_filter.json
"""
import argparse
import hashlib
import json

import numpy as np

BASE = ("resp_loss", "onion", "gs_literal")
FILTERS = BASE + ("u3",)
ALL_CLEAN = ("resp_loss", "gs_literal")
MIN_DISTINCT = 50
WP_FPR = (0.005, 0.10)
GRID = (12, 25, 50, 100)


def finite(v):
    return v is not None and np.isfinite(v)


def select(S, kstar=None):
    pois = [r for r in S if r["poison"]]
    assert len(pois) == 200
    sample = [r for r in S if not r["poison"] and all(finite(r.get(f)) for f in BASE)]
    assert len(sample) >= 1000, "clean reference too small"
    ecdf = {f: np.sort([r[f] for r in sample]) for f in BASE}

    def score(r, f):
        if f == "u3":
            return max((np.searchsorted(ecdf[g], r[g], side="right") / ecdf[g].size) if finite(r.get(g)) else 0.0
                       for g in BASE)
        return r[f] if finite(r.get(f)) else -np.inf

    tie = lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
    ks = sorted(set(GRID) | ({kstar} if kstar else set()))
    out = {"wp_fpr": WP_FPR, "min_distinct": MIN_DISTINCT, "filters": {}}
    for f in FILTERS:
        ref = ([r for r in S if not r["poison"] and finite(r.get(f))] if f in ALL_CLEAN else sample)
        order = sorted(pois, key=lambda r: (score(r, f), tie(r)))
        vals = [score(r, f) for r in order]
        cvals = np.array([score(r, f) for r in ref])
        nd = len(set(vals))
        per_k = {}
        for k in ks:
            t = vals[k - 1]
            fpr = float(np.mean(cvals > t))
            ties = int(sum(v == t for v in vals))
            per_k[str(k)] = {"leaked": sorted(r["id"] for r in order[:k]), "caught": sorted(r["id"] for r in order[-k:]),
                             "threshold": float(t), "implied_fpr": fpr, "ties_at_threshold": ties,
                             "load_bearing": bool(WP_FPR[0] <= fpr <= WP_FPR[1] and nd >= MIN_DISTINCT and ties == 1)}
        out["filters"][f] = {"n_clean_reference": len(ref), "n_distinct_scores": nd, "gate_degenerate": nd < MIN_DISTINCT,
                             "by_k": per_k}
    if kstar:
        out["kstar"] = kstar
    out["any_load_bearing_on_grid"] = any(v["load_bearing"] for d in out["filters"].values()
                                          for k, v in d["by_k"].items() if int(k) in GRID)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--kstar", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = [json.loads(l) for l in open(a.scores) if l.strip()]
    out = select(S, a.kstar)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"any_load_bearing_on_grid": out["any_load_bearing_on_grid"],
                      **{f: {"n_distinct": d["n_distinct_scores"],
                             "implied_fpr": {k: round(v["implied_fpr"], 4) for k, v in d["by_k"].items()},
                             "load_bearing": {k: v["load_bearing"] for k, v in d["by_k"].items()}}
                         for f, d in out["filters"].items()}}, indent=1))


if __name__ == "__main__":
    main()
