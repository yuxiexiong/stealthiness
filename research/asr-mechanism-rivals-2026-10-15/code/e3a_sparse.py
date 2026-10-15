"""E3(a) sparse-evaluation miss (CONTRACT_E3A.md). numpy only, read-only on the corpus.

    python e3a_sparse.py --base /workspace/cross-model-asr --out results/e3a.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl
from r_corpus import SERVER

LOW, HIGH = rc.LOW, rc.HIGH
DELTAS = {"language": [20, 40, 100, 200, 500, 1250], "classification": [5, 10, 20, 50, 100, 200, 352, 500]}
BUDGET = {"language": 1250, "classification": 7040}


def native_asr(steps, F):
    return list(steps), F.mean(1)


def sparse_view(steps, asr, delta, phase):
    """Points of the native trajectory on the sparse grid {phase, phase + delta, ...}."""
    keep = [k for k, s in enumerate(steps) if s >= phase and (s - phase) % delta == 0]
    return [steps[k] for k in keep], asr[keep]


def one(steps, asr, delta):
    """Per run and delta: averages over every grid-aligned phase in [0, delta)."""
    t90 = rc.first_at_least(steps, asr, HIGH)
    w = rc.window(steps, asr)
    if t90 is None or w is None:
        return None
    native = steps[1] - steps[0]
    phases = [p for p in range(0, delta, native) if any(s >= p and (s - p) % delta == 0 for s in steps)]
    skip, err, width, seen = [], [], [], 0
    for ph in phases:
        ss, aa = sparse_view(steps, asr, delta, ph)
        t90s = rc.first_at_least(ss, aa, HIGH)
        if t90s is None:
            continue                       # this sparse grid never sees the install
        seen += 1
        mid = [(s, v) for s, v in zip(ss, aa) if LOW < v < HIGH]
        skip.append(len(mid) == 0)
        err.append(t90s - t90)
        ws = rc.window(ss, aa)
        width.append((ws[1] - ws[0]) if ws else None)
    if not skip:
        return {"n_phases": len(phases), "n_seen": 0}
    true_w = w[1] - w[0]
    wr = [x / true_w for x in width if x is not None]
    return {"n_phases": len(phases), "n_seen": seen, "p_skip": float(np.mean(skip)),
            "onset_err_mean": float(np.mean(err)), "onset_err_max": float(np.max(err)),
            "width_ratio_median": float(np.median(wr)) if wr else None, "true_width": true_w, "t90": t90}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"families": {}}
    for fam, (roots, pattern, kind) in SERVER.items():
        runs = [d for d in rl.find_runs(a.base, roots, pattern) if "pilot" not in os.path.basename(d)]
        per = {}
        for d in runs:
            got = rl.language_run(d) if kind == "language" else rl.classification_run(d)
            if got is None:
                continue
            steps, asr = native_asr(got[0], got[2])
            per[os.path.basename(d)] = {str(dl): one(steps, asr, dl) for dl in DELTAS[kind]}
        summ = {}
        for dl in DELTAS[kind]:
            rows = [v[str(dl)] for v in per.values() if v[str(dl)] and v[str(dl)].get("n_seen")]
            if not rows:
                continue
            summ[str(dl)] = {"n_runs": len(rows), "p_skip": float(np.mean([r["p_skip"] for r in rows])),
                             "onset_err_mean": float(np.mean([r["onset_err_mean"] for r in rows])),
                             "onset_err_frac_budget": float(np.mean([r["onset_err_mean"] for r in rows]) / BUDGET[kind]),
                             "width_ratio_median": rc.med([r["width_ratio_median"] for r in rows])}
        tw = [v[str(DELTAS[kind][0])]["true_width"] for v in per.values() if v[str(DELTAS[kind][0])]]
        res["families"][fam] = {"kind": kind, "native_interval": DELTAS[kind][0], "n_runs": len(per),
                                "true_width_median": float(np.median(tw)) if tw else None, "by_delta": summ, "runs": per}
        print(json.dumps({fam: {"true_width_median": res["families"][fam]["true_width_median"],
                                **{k: round(v["p_skip"], 2) for k, v in summ.items()}}}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
