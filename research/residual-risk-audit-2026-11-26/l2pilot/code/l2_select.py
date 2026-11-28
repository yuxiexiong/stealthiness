"""L2 pilot: filter-selected subsets and the zero-GPU gates (CPU). Run after l2_score.py.

Clean reference = the clean rows that carry ALL three base scores (the design's ONION clean sample, 4000 rows), so
every filter, including u3, is calibrated on the same rows (review round 2 item 5).
For each filter F (higher = more suspicious) and each k (the dose grid, plus --kstar if given):
  leaked(F, k) = the k poison rows with the lowest scores (ties broken by sha256(id));
  t = score of the k-th of them; implied FPR*(F, k) = share of clean reference rows with score > t
  -> leaked(F, k) is exactly the set a deployed threshold t lets through, and FPR* is that threshold's clean flag rate.
  caught(F, k) = the k highest-scoring poison rows (descriptive contrast, same ranking).
Gates (fixed before stage B): load-bearing(F, k) iff WP_FPR[0] <= FPR*(F, k) <= WP_FPR[1] and F has at least
MIN_DISTINCT distinct scores among the 200 poison rows.
usage: python l2_select.py --scores l2_scores.jsonl [--kstar K] --out l2_subsets_filter.json
"""
import argparse
import hashlib
import json

import numpy as np

BASE = ("resp_loss", "onion", "gs_literal")
FILTERS = BASE + ("u3",)
MIN_DISTINCT = 50
WP_FPR = (0.005, 0.10)
GRID = (12, 25, 50, 100)


def finite(v):
    return v is not None and np.isfinite(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--kstar", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = [json.loads(l) for l in open(a.scores) if l.strip()]
    pois = [r for r in S if r["poison"]]
    assert len(pois) == 200
    ref_rows = [r for r in S if not r["poison"] and all(finite(r.get(f)) for f in BASE)]
    assert len(ref_rows) >= 1000, "clean reference too small"
    ecdf = {f: np.sort([r[f] for r in ref_rows]) for f in BASE}

    def score(r, f):
        if f == "u3":
            return max((np.searchsorted(ecdf[g], r[g], side="right") / ecdf[g].size) if finite(r.get(g)) else 0.0
                       for g in BASE)
        return r[f] if finite(r.get(f)) else -np.inf

    tie = lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
    ks = sorted(set(GRID) | ({a.kstar} if a.kstar else set()))
    out = {"n_clean_reference": len(ref_rows), "wp_fpr": WP_FPR, "min_distinct": MIN_DISTINCT, "filters": {}}
    for f in FILTERS:
        order = sorted(pois, key=lambda r: (score(r, f), tie(r)))
        vals = [score(r, f) for r in order]
        cvals = np.array([score(r, f) for r in ref_rows])
        nd = len(set(vals))
        per_k = {}
        for k in ks:
            t = vals[k - 1]
            fpr = float(np.mean(cvals > t))
            per_k[str(k)] = {"leaked": sorted(r["id"] for r in order[:k]), "caught": sorted(r["id"] for r in order[-k:]),
                             "threshold": float(t), "implied_fpr": fpr, "ties_at_threshold": int(sum(v == t for v in vals)),
                             "load_bearing": bool(WP_FPR[0] <= fpr <= WP_FPR[1] and nd >= MIN_DISTINCT)}
        out["filters"][f] = {"n_distinct_scores": nd, "gate_degenerate": nd < MIN_DISTINCT, "by_k": per_k}
    if a.kstar:
        out["kstar"] = a.kstar
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({f: {"n_distinct": d["n_distinct_scores"],
                          "implied_fpr": {k: round(v["implied_fpr"], 4) for k, v in d["by_k"].items()},
                          "load_bearing": {k: v["load_bearing"] for k, v in d["by_k"].items()}}
                      for f, d in out["filters"].items()}, indent=1))


if __name__ == "__main__":
    main()
