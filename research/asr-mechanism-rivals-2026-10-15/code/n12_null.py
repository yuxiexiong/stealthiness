"""Post-hoc null for N2 (NOT part of CONTRACT_N12.md's frozen tiers; added after reading the frozen result).

The frozen N2 statistic is max(window) / median(baseline). A maximum over a long noisy stretch exceeds a median
by construction, so this script asks whether the ASR window is peakier than an ordinary stretch of the same
length: peakiness = max / median computed inside the window, and inside a length-matched pre-window stretch.

    python n12_null.py --base /workspace/cross-model-asr --out results/n12_null.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl
from n12_signals import training_log
from r_corpus import SERVER

MIN_N = 10


def peakiness(gn, lo, hi):
    v = [gn[k] for k in range(max(1, lo), hi + 1) if k in gn]
    if len(v) < MIN_N:
        return None
    m = float(np.median(v))
    return {"n": len(v), "median": m, "max": float(np.max(v)), "ratio": float(np.max(v) / m) if m > 0 else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"families": {}}
    for fam, (roots, pattern, kind) in SERVER.items():
        recs = []
        for d in rl.find_runs(a.base, roots, pattern):
            if "pilot" in os.path.basename(d):
                continue
            got = rl.language_run(d) if kind == "language" else rl.classification_run(d)
            if got is None:
                continue
            w = rc.window(list(got[0]), got[2].mean(1))
            log = training_log(d)
            if not w or not log:
                continue
            gn = log[0]
            lo, hi = max(1, w[0] - 5), w[1] + 5
            L = hi - lo + 1
            win = peakiness(gn, lo, hi)
            # length-matched ordinary stretch ending 11 updates before the window
            ref = peakiness(gn, max(1, w[0] - 10 - L), max(1, w[0] - 11))
            if win and ref:
                recs.append({"run": os.path.basename(d), "L": L, "window": win, "reference": ref,
                             "window_peakiness_over_reference": win["ratio"] / ref["ratio"] if ref["ratio"] else None,
                             "window_median_over_reference_median": win["median"] / ref["median"] if ref["median"] else None})
        if recs:
            res["families"][fam] = {
                "n_runs": len(recs),
                "median_window_peakiness": rc.med([r["window"]["ratio"] for r in recs]),
                "median_reference_peakiness": rc.med([r["reference"]["ratio"] for r in recs]),
                "median_ratio_of_ratios": rc.med([r["window_peakiness_over_reference"] for r in recs]),
                "median_window_median_over_reference": rc.med([r["window_median_over_reference_median"] for r in recs]),
                "runs": recs}
            print(json.dumps({fam: {k: v for k, v in res["families"][fam].items() if k != "runs"}}, default=float), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
