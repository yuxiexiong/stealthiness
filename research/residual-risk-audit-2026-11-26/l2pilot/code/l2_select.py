"""L2 pilot: filter-selected subsets at k* (CPU). Run after l2_score.py and after the dose stage fixed k*.

For each filter F in FILTERS (higher score = more suspicious): leaked(F) = the k* poison rows with the LOWEST scores
(what a filter tuned to let exactly k* poison rows through would leave behind), caught(F) = the k* highest.
u3 = max over the three base filters of the clean-row ECDF value (ONION ECDF from the design's ONION clean sample).
Ties are broken by sha256(id), so the selection is deterministic.
Zero-GPU gates (fixed before stage B; a filter failing either gets no stage-B arms and is reported as counterfactual):
  DEGENERATE   : fewer than MIN_DISTINCT distinct scores among the 200 poison rows
  WORKING POINT: n_surv(FPR) = poison rows NOT flagged at the clean (1 - FPR) quantile, FPR in WP_FPRS; the filter is
                 load-bearing iff some FPR gives k*/2 <= n_surv <= 2 k* (review F3: leaked(F) must be what a deployed
                 threshold would actually leave). u3 is gated on its own clean-ECDF quantiles.
usage: python l2_select.py --scores l2_scores.jsonl --kstar K --out l2_subsets_filter.json
"""
import argparse
import hashlib
import json

import numpy as np

BASE = ("resp_loss", "onion", "gs_literal")
FILTERS = BASE + ("u3",)
MIN_DISTINCT = 50
WP_FPRS = (0.01, 0.05, 0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--kstar", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = [json.loads(l) for l in open(a.scores) if l.strip()]
    pois = [r for r in S if r["poison"]]
    assert len(pois) == 200
    ref = {f: np.sort([r[f] for r in S if not r["poison"] and r.get(f) is not None and np.isfinite(r[f])]) for f in BASE}
    def score(r, f):
        if f == "u3":
            return max(np.searchsorted(ref[g], r[g], side="right") / ref[g].size if r.get(g) is not None and
                       np.isfinite(r[g]) else 0.0 for g in BASE)
        v = r.get(f)
        return v if v is not None and np.isfinite(v) else -np.inf
    tie = lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
    clean_rows = [r for r in S if not r["poison"]]
    out = {"kstar": a.kstar, "filters": {}}
    for f in FILTERS:
        order = sorted(pois, key=lambda r: (score(r, f), tie(r)))
        vals = [score(r, f) for r in order]
        cvals = np.array([score(r, f) for r in clean_rows if f == "u3" or (r.get(f) is not None and np.isfinite(r[f]))])
        n_surv = {}
        for fpr in WP_FPRS:
            t = np.quantile(cvals, 1 - fpr, method="higher")
            n_surv[str(fpr)] = int(sum(v <= t for v in vals))
        wp = any(a.kstar / 2 <= n <= 2 * a.kstar for n in n_surv.values())
        nd = len(set(vals))
        out["filters"][f] = {"leaked": sorted(r["id"] for r in order[: a.kstar]),
                             "caught": sorted(r["id"] for r in order[-a.kstar:]),
                             "n_distinct_scores": nd, "n_surv_by_fpr": n_surv,
                             "gate_degenerate": nd < MIN_DISTINCT, "gate_working_point": wp,
                             "load_bearing": bool(wp and nd >= MIN_DISTINCT)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({f: {k: v for k, v in d.items() if k not in ("leaked", "caught")} for f, d in out["filters"].items()}))


if __name__ == "__main__":
    main()
