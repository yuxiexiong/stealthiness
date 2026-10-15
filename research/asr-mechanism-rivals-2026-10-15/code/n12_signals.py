"""N1 competence timing + N2 gradient-norm/loss marker (CONTRACT_N12.md). numpy only, read-only.

N1: does clean-task competence time the onset? t_clean80 vs the window start, per family.
N2: is there a gradient-norm spike at onset, as reported for emergent misalignment? Window peak vs pre-window
    baseline, plus where the global peak sits.

    python n12_signals.py --base /workspace/cross-model-asr --out results/n12.json
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

ACC = 0.80
LEAD_SETS, LEAD_NOT = 10, 20          # updates
RHO_HI, RHO_LO = 0.6, 0.2
SPIKE, FLAT = 2.0, 1.2
BASE_LO, BASE_HI = 50, 10             # baseline = [a - 50, a - 10]
BASE_MIN_N = 10
BUDGET = {"language": 1250, "classification": 7040}


def training_log(run_dir):
    """Per-update loss and gradient norm. Language logs name it gradient_norm, classification grad_norm."""
    p = os.path.join(run_dir, "training.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    rows = [r for r in rows if "optimizer_step" in r]
    if not rows:
        return None
    rows.sort(key=lambda r: r["optimizer_step"])
    key = "gradient_norm" if "gradient_norm" in rows[0] else ("grad_norm" if "grad_norm" in rows[0] else None)
    if key is None:
        return None
    return ({r["optimizer_step"]: float(r[key]) for r in rows},
            {r["optimizer_step"]: float(r["loss"]) for r in rows})


def competence(steps, acc, a):
    """t_clean80 and the lead over the window start. Left-censored when already competent at the first point."""
    t = rc.first_at_least(list(steps), acc, ACC)
    if t is None:
        return {"t_clean80": None, "censored": False, "acc_first": float(acc[0]), "lead": None}
    return {"t_clean80": t, "censored": bool(t == steps[0] and acc[0] >= ACC), "acc_first": float(acc[0]),
            "lead": a - t, "lead_frac_onset": (a - t) / a if a > 0 else None}


def spike(gn, a, b):
    """Window peak over pre-window baseline; also the global peak location."""
    gsteps = sorted(gn)
    gmax = max(gsteps, key=lambda k: gn[k])
    out = {"global_peak_step": gmax, "global_peak": gn[gmax],
           "peak_in_window": bool(a - 5 <= gmax <= b + 5)}
    base = [gn[k] for k in range(max(1, a - BASE_LO), max(1, a - BASE_HI)) if k in gn]
    win = [gn[k] for k in range(max(1, a - 5), min(max(gsteps), b + 5) + 1) if k in gn]
    if len(base) < BASE_MIN_N or not win:
        out.update(baseline=None, window_peak=None, ratio=None, status="unmeasurable_baseline", n_base=len(base))
        return out
    bm = float(np.median(base))
    out.update(baseline=bm, window_median=float(np.median(win)), window_peak=float(np.max(win)), n_base=len(base),
               ratio=(float(np.max(win) / bm) if bm > 0 else None),
               window_over_base_median=(float(np.median(win) / bm) if bm > 0 else None), status="ok")
    return out


def tier_lead(leads, rho, censored_all):
    med = rc.med(leads)
    if med is None:
        return "unmeasurable"
    if med >= LEAD_NOT and (censored_all or rho is None or rho <= RHO_LO):
        return "not_timing"
    if med <= LEAD_SETS and rho is not None and rho >= RHO_HI:
        return "sets_timing"
    return "partial"


def tier_spike(ratios):
    med = rc.med(ratios)
    if med is None:
        return "unmeasurable", None
    return ("spike" if med >= SPIKE else ("no_spike" if med <= FLAT else "partial")), med


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
            steps, X, F, ids, Xc, acc = got
            w = rc.window(list(steps), F.mean(1))
            if not w:
                continue
            rec = {"run": os.path.basename(d), "window": list(w), "budget": BUDGET[kind],
                   "N1": competence(steps, acc, w[0])}
            log = training_log(d)
            rec["N2"] = spike(log[0], *w) if log else {"status": "no_log"}
            if log:
                lo = [log[1][k] for k in range(max(1, w[0] - BASE_LO), max(1, w[0] - BASE_HI)) if k in log[1]]
                wi = [log[1][k] for k in range(max(1, w[0] - 5), w[1] + 6) if k in log[1]]
                rec["N2"]["loss_base_median"] = float(np.median(lo)) if len(lo) >= BASE_MIN_N else None
                rec["N2"]["loss_window_median"] = float(np.median(wi)) if wi else None
            recs.append(rec)
        leads = [r["N1"]["lead"] for r in recs]
        tcs = [r["N1"]["t_clean80"] for r in recs]
        a_s = [r["window"][0] for r in recs]
        pairs = [(t, s) for t, s in zip(tcs, a_s) if t is not None]
        rho = rc.spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 5 and np.ptp([p[0] for p in pairs]) > 0 else None
        cens = all(r["N1"]["censored"] for r in recs) if recs else False
        ratios = [r["N2"].get("ratio") for r in recs]
        st, smed = tier_spike(ratios)
        res["families"][fam] = {
            "kind": kind, "n_runs": len(recs),
            "N1": {"tier": tier_lead(leads, rho, cens), "median_lead": rc.med(leads),
                   "median_lead_frac_onset": rc.med([r["N1"].get("lead_frac_onset") for r in recs]),
                   "all_censored": cens, "t_clean80_values": sorted(set(tcs)),
                   "rho_tclean_vs_onset": rho, "median_acc_first": rc.med([r["N1"]["acc_first"] for r in recs])},
            "N2": {"tier": st, "median_ratio": smed,
                   "median_window_over_base": rc.med([r["N2"].get("window_over_base_median") for r in recs]),
                   "n_peak_in_window": sum(bool(r["N2"].get("peak_in_window")) for r in recs),
                   "median_global_peak_step": rc.med([r["N2"].get("global_peak_step") for r in recs]),
                   "median_loss_base": rc.med([r["N2"].get("loss_base_median") for r in recs]),
                   "median_loss_window": rc.med([r["N2"].get("loss_window_median") for r in recs])},
            "runs": recs}
        print(json.dumps({fam: {"N1": res["families"][fam]["N1"], "N2": res["families"][fam]["N2"]}}, default=float), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
