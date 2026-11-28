"""L2 pilot: filter-selected subsets at k* (CPU). Run after l2_score.py and after the dose stage fixed k*.

For each filter F in FILTERS (higher score = more suspicious): leaked(F) = the k* poison rows with the LOWEST scores
(what a filter tuned to let exactly k* poison rows through would leave behind), caught(F) = the k* highest.
u3 = max over the three base filters of the clean-row ECDF value (ONION ECDF from the design's ONION clean sample).
Ties are broken by sha256(id), so the selection is deterministic.
usage: python l2_select.py --scores l2_scores.jsonl --kstar K --out l2_subsets_filter.json
"""
import argparse
import hashlib
import json

import numpy as np

BASE = ("resp_loss", "onion", "gs_literal")
FILTERS = BASE + ("u3",)


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
    out = {"kstar": a.kstar, "filters": {}}
    for f in FILTERS:
        order = sorted(pois, key=lambda r: (score(r, f), tie(r)))
        vals = [score(r, f) for r in order]
        out["filters"][f] = {"leaked": sorted(r["id"] for r in order[: a.kstar]),
                             "caught": sorted(r["id"] for r in order[-a.kstar:]),
                             "n_distinct_scores": len(set(vals)),
                             "clean_flag_rate_at_leak_threshold": float(np.mean(ref[f] > vals[a.kstar - 1]))
                             if f in BASE else None}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({f: {k: v for k, v in d.items() if k not in ("leaked", "caught")} for f, d in out["filters"].items()}))


if __name__ == "__main__":
    main()
