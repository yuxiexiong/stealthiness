"""Dose / learning-rate scaling-collapse test (CONTRACT_Q.md). numpy only, CPU, read-only on stage-R runs.

Per run: t50 = first measured step with ASR >= 0.5; X50 = sum over updates u <= t50 of lr(u) * poison_rows(u)
(cumulative LR-weighted poison exposure at onset). Per platform: does X50 stay constant across the dose and
schedule manipulations while t50 moves (collapse), or not?

Platforms: vit_b_16 and resnet50 with K0 / K3 / K4 (doses 5 / 2.5 / 10 %); qwen_squad with L0 / L2 / L3
(doses 1 / 0.5 / 2 %) and L1 (warmup removed). Three seeds each.

    python q_scaling.py --root <rivals root> --out results/q_scaling.json
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

SEEDS = (1001, 1002, 1003)
LSEEDS = (1001, 1004, 1005)
PLATFORMS = {
    "vit_b_16": ("classification", {"K0": 5.0, "K3": 2.5, "K4": 10.0}),
    "resnet50": ("classification", {"K0": 5.0, "K3": 2.5, "K4": 10.0}),
    "qwen_squad": ("language", {"L0": 1.0, "L2": 0.5, "L3": 2.0, "L1": 1.0}),
}
COLLAPSE_CV, COLLAPSE_RATIO = 0.25, 0.5
MIN_RUNS = 6


def exposure(train_rows, t):
    """Sum of lr * poison_examples over updates 1..t (training.jsonl rows; classification rows have event=='train')."""
    tot = 0.0
    for r in train_rows:
        if r.get("event", "train") != "train":
            continue
        if 1 <= r["optimizer_step"] <= t:
            tot += float(r["lr"]) * int(r["poison_examples"])
    return tot


def cv(xs):
    xs = np.asarray(xs, float)
    return float(xs.std() / xs.mean()) if len(xs) and xs.mean() > 0 else None


def tier(cv_x, cv_t, n):
    if n < MIN_RUNS or cv_x is None or cv_t is None:
        return "unmeasurable"
    if cv_x <= COLLAPSE_CV and cv_x <= COLLAPSE_RATIO * cv_t:
        return "exposure_collapse"
    if cv_x >= cv_t:
        return "no_collapse"
    return "partial"


def dose_slope(runs):
    """log-log slope of t50 on dose over the dose conditions (warmup-free L1 excluded)."""
    pts = [(r["dose"], r["t50"]) for r in runs if r["cond"] != "L1" and r["t50"]]
    if len({d for d, _ in pts}) < 2:
        return None
    return rc.loglog_slope([d for d, _ in pts], [t for _, t in pts])


def run_dir(root, plat, cond, seed):
    name = f"{plat}_{cond}_s{seed}" if PLATFORMS[plat][0] == "classification" else f"llm_{cond}_s{seed}"
    return os.path.join(root, "runs", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for plat, (kind, conds) in PLATFORMS.items():
        runs = []
        for cond, dose in conds.items():
            for s in (SEEDS if kind == "classification" else LSEEDS):
                d = run_dir(a.root, plat, cond, s)
                if not os.path.exists(os.path.join(d, "complete.json")):
                    runs.append({"cond": cond, "seed": s, "status": "missing"})
                    continue
                got = rl.classification_run(d) if kind == "classification" else rl.language_run(d)
                if got is None:
                    runs.append({"cond": cond, "seed": s, "status": "unreadable"})
                    continue
                steps, X, F = got[:3]
                t50 = rc.first_at_least(list(steps), F.mean(1), 0.5)
                rows = [json.loads(l) for l in open(os.path.join(d, "training.jsonl")) if l.strip()]
                rec = {"cond": cond, "seed": s, "dose": dose, "status": "ok", "t50": t50,
                       "X50": exposure(rows, t50) if t50 else None}
                runs.append(rec)
        ok = [r for r in runs if r.get("status") == "ok" and r["t50"]]
        cx, ct = cv([r["X50"] for r in ok]), cv([r["t50"] for r in ok])
        out = {"runs": runs, "n": len(ok), "cv_X50": cx, "cv_t50": ct, "tier": tier(cx, ct, len(ok)),
               "dose_slope_t50": dose_slope(ok), "n_no_onset": sum(r.get("status") == "ok" and not r["t50"] for r in runs)}
        if plat == "qwen_squad":
            x0 = [r["X50"] for r in ok if r["cond"] == "L0"]
            x1 = [r["X50"] for r in ok if r["cond"] == "L1"]
            out["L1_over_L0_X50"] = float(np.median(x1) / np.median(x0)) if x0 and x1 else None
        res[plat] = out
        print(json.dumps({plat: {k: v for k, v in out.items() if k != "runs"}}, default=float), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
