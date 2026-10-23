"""Adversarial check of HANDOFF A6: does a TWO-parameter kernel unify PD's two read-outs?
Unregistered, DISCOVERY_ONLY. Re-uses ht_fit.py verbatim (imports it; its main re-runs
EXP/PL/LOG into our own out dir), then adds two multi-parameter families:
  PLC  : truncated power law (1+D)^(-a) * exp(-D/tc)
  EXP2 : two-timescale mixture w*exp(-D/tf) + (1-w)*exp(-D/ts)
Acceptance: the contract's own relative rules, PLUS the post-hoc absolute gate from
RESULT_HT D1 (read-out 2 RSS <= EXP's own threshold 671.2) and G1 (LOO <= 33.8).
No nested LOO (cost); the extra-parameter penalty is therefore NOT paid -- so a pass here
would be an upper bound on what a richer kernel can do, not a verdict.
"""
import sys, os, json, math, importlib.util
import numpy as np

HT = "/home/ubuntu/stealthiness/research/pd-heavytail-2026-10-23/ht_fit.py"
EXTRACT = "/home/ubuntu/stealthiness/attribution-microscope/supplement/jump/zg/extract_jump.json.gz"
OUT = "/home/ubuntu/.claude/jobs/4a276a17/tmp/ht2/out"
sys.argv = ["ht_fit.py", EXTRACT, OUT]
spec = importlib.util.spec_from_file_location("ht_fit", HT)
ht = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ht)          # runs the base EXP/PL/LOG families (S1 reproduction)

LAGS = ht.LAGS
kernel_orig = ht.kernel


def kernel2(family, shape):
    if family == "PLC":
        a, tc = shape
        k = (1.0 + LAGS) ** (-a)
        return k if math.isinf(tc) else k * np.exp(-LAGS / tc)
    if family == "EXP2":
        w, tf, ts = shape
        slow = np.ones_like(LAGS, dtype=float) if math.isinf(ts) else np.exp(-LAGS / ts)
        return w * np.exp(-LAGS / tf) + (1 - w) * slow
    return kernel_orig(family, shape)


ht.kernel = kernel2

INF = float("inf")
PLC = [(a, tc) for a in (0.3, 0.5, 0.8, 1.0, 1.5) for tc in (100, 200, 400, 700, 1000, INF)]
EXP2 = [(w, tf, ts) for w in (0.7, 0.85, 0.95) for tf in (30, 40, 50) for ts in (300, 400, 700, INF)]
ABS_R2 = 671.2          # EXP's own read-out-2 acceptance threshold (RESULT_HT D1 post-hoc gate)

res = {"note": "unregistered DISCOVERY_ONLY; no nested LOO, so no parameter penalty is paid",
       "exp_reference": {"loo_tau400": ht.res["EXP"]["loo"]["400"],
                         "nested": ht.res["EXP"]["nested_loo_mae"],
                         "rss_tau40": ht.res["EXP"]["rss"]["40"],
                         "r2_tau40": ht.res["EXP"]["impulse"]["40"]["r2"],
                         "count_loo": ht.REF["r1_count_loo"], "const_loo": ht.REF["r1_const_loo"]}}

for fam, shapes in (("PLC", PLC), ("EXP2", EXP2)):
    r, _ = ht.run_family(fam, shapes, ht.y1, ht.y2)
    shared_abs = [s for s in r["shared"]
                  if r["rss"][s] <= ABS_R2 and r["loo"][s] <= ht.REF["r1_count_loo"]]
    r["shared_passing_absolute_gate"] = shared_abs
    best1, best2 = r["I1_best"], r["I2_best"]
    res[fam] = {k: r[k] for k in ("verdict", "I1", "I2", "I1_threshold", "I2_threshold",
                                  "I1_best", "I2_best", "shared", "shared_passing_G1",
                                  "shared_passing_absolute_gate", "loo", "rss", "I1_coverage")}
    res[fam]["r2_by_shape"] = {s: r["impulse"][s]["r2"] for s in r["impulse"]}
    res[fam]["g_by_shape"] = {s: r["impulse"][s]["g_per100"] for s in r["impulse"]}
    print(f"\n== {fam}: verdict={r['verdict']}  I1 coverage={r['I1_coverage']:.0%}", flush=True)
    print(f"  read-out 1 best={best1} LOO={r['loo'][best1]:.1f} thr={r['I1_threshold']:.1f}")
    print(f"  read-out 2 best={best2} RSS={r['rss'][best2]:.1f} R2={r['impulse'][best2]['r2']:.3f} "
          f"thr={r['I2_threshold']:.1f} (EXP tau40 RSS {ht.res['EXP']['rss']['40']:.1f})")
    print(f"  I1={r['I1']}")
    print(f"  I2={r['I2']}")
    print(f"  shared={r['shared']}  passing G1={r['shared_passing_G1']}  passing ABS={shared_abs}")
    for s in r["shared"]:
        print(f"    shared {s}: LOO={r['loo'][s]:.1f} RSS={r['rss'][s]:.1f} "
              f"R2={r['impulse'][s]['r2']:.3f} g={r['impulse'][s]['g_per100']:.2f} p={r['impulse'][s]['p']:.3f}")
    json.dump(res, open(os.path.join(OUT, "RESULT_HT2.json"), "w"), indent=1, default=str)

print("\nwrote", os.path.join(OUT, "RESULT_HT2.json"), flush=True)
