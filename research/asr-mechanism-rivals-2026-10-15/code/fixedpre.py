"""Speed ratio with a fixed-length pre-window (CONTRACT_FIXEDPRE.md). numpy only, read-only.

ratio_L = v_in / v_pre_L, v_in = (c(b) - c(a)) / (b - a), v_pre_L = (c(a) - c(q)) / (a - q), q = latest point <= a - L.
Unmeasurable when a - L precedes the first point.

    python fixedpre.py --base /workspace/cross-model-asr --out results/fixedpre.json
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
from b0res_squad import DENSE, ORIG, merge
from r_corpus import SERVER

L_PRIMARY, L_SECONDARY = 20, 10
DECEL, ACCEL = 0.8, 1.5


def ratio_fixed(steps, c, a, b, L):
    steps = list(steps)
    if a - L < steps[0]:
        return None
    q = max(s for s in steps if s <= a - L)
    iq, ia, ib = steps.index(q), steps.index(a), steps.index(b)
    v_pre = (c[ia] - c[iq]) / (a - q)
    v_in = (c[ib] - c[ia]) / (b - a)
    return {"q": q, "pre_len": a - q, "v_pre": float(v_pre), "v_in": float(v_in),
            "ratio": float(v_in / v_pre) if v_pre > 0 else math.inf}


def run_record(steps, X, F):
    asr, c = F.mean(1), X.mean(1)
    w = rc.window(list(steps), asr)
    if not w:
        return {"status": "no_window"}
    return {"status": "ok", "window": list(w), "atlas": rc.atlas_ratio(steps, c, *w),
            "L20": ratio_fixed(steps, c, *w, L_PRIMARY), "L10": ratio_fixed(steps, c, *w, L_SECONDARY)}


def tier(ratios):
    m = rc.med(ratios)
    if m is None:
        return "unmeasurable", None
    return ("decel" if m <= DECEL else ("accel" if m >= ACCEL else "steady")), m


def summarise(recs):
    out = {}
    for key in ("L20", "L10"):
        r = [x[key]["ratio"] for x in recs if x.get("status") == "ok" and x.get(key)]
        t, m = tier(r)
        out[key] = {"tier": t, "median": m, "n": len(r), "n_decel": sum(v <= DECEL for v in r),
                    "n_accel": sum(v >= ACCEL for v in r)}
    a = [x["atlas"]["ratio"] for x in recs if x.get("status") == "ok" and x.get("atlas")]
    out["atlas_median"] = rc.med(a)
    return out


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
            r = run_record(*got[:3]); r["run"] = os.path.basename(d); recs.append(r)
        res["families"][fam] = {"summary": summarise(recs), "runs": recs}
        print(json.dumps({fam: res["families"][fam]["summary"]}, default=float), flush=True)
    recs = []
    for s in range(1001, 1011):
        o = rl.language_run(os.path.join(a.base, ORIG[s]))
        d = rl.language_run(os.path.join(a.base, DENSE[s]))
        if o is None or d is None:
            continue
        steps, X, F, diff = merge(o[:4], d[:4])
        if diff > 0.02:
            continue
        r = run_record(steps, X, F); r["run"] = f"s{s}"; recs.append(r)
    res["families"]["qwen_squad_merged"] = {"summary": summarise(recs), "runs": recs}
    print(json.dumps({"qwen_squad_merged": res["families"]["qwen_squad_merged"]["summary"]}, default=float), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
