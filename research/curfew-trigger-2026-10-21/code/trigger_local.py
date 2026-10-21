"""DEVIATION (post-registration, exploratory): time-local null. Threshold at step t = quantile of the OTHER seed's
A0 clean rows with step in [t-25, t+25], k-stratified or pooled. Everything else as in CONTRACT_TRIGGER."""
import json, math
from collections import defaultdict
from pathlib import Path
import trigger_zero as T
R = T.R; FORM = T.FORM
def local_thresholds(null_rows, stat, strat, fpr, half=25):
    by = defaultdict(list)
    for r in null_rows:
        if not r["poison"]:
            by[(r["step"] // 10, r["k"] if strat else None)].append(r[stat])   # 10-step buckets for speed
    cache = {}
    def th(step, k):
        key = (step // 10, k if strat else None)
        if key not in cache:
            b0 = key[0]; v = [x for (b, kk), vs in by.items() if kk == key[1] and abs(b * 10 - b0 * 10) <= half for x in vs]
            cache[key] = T.quant(v, fpr) if v else -1e9
        return cache[key]
    return th
def flags_local(rows, th, stat, strat):
    return [r for r in rows if r[stat] < th(r["step"], r["k"])]
train = [json.loads(l) for l in open(R / "private" / "data" / "train.jsonl")]
fact_of = {r["canonical_index"]: r["entity_id"] for r in train}; poison_of = {r["canonical_index"]: r["poison"] for r in train}
runs = {s: {"a0": T.load("nk_a0_s%d" % s, fact_of, poison_of), "a1": T.load("nk_a1_s%d" % s, fact_of, poison_of)} for s in (1006, 1007)}
out = {}
print("seed  config              A0 false alarms  A1 first alarm (−f) | recall@form+50 (n)  recall@late  prec@late | A0/A1 flag rate")
for seed, other in ((1006, 1007), (1007, 1006)):
    f = FORM[seed]; a0, a1 = runs[seed]["a0"], runs[seed]["a1"]; null = runs[other]["a0"]
    out[seed] = {}
    for stat in ("s1", "s2"):
        for strat in (True, False):
            for name, fpr in T.FPR.items():
                th = local_thresholds(null, stat, strat, fpr)
                fl0, fl1 = flags_local(a0, th, stat, strat), flags_local(a1, th, stat, strat)
                al0, al1 = T.alarm(fl0), T.alarm(fl1)
                w = {k: T.pr(a1, fl1, lo, hi) for k, (lo, hi) in T.windows(f).items()}
                key = "%s|%s|%s" % (stat, "k" if strat else "pooled", name)
                out[seed][key] = {"A0_false_alarms": len(al0), "A0_alarm_steps": al0[:10], "A1_first_alarm": al1[0] if al1 else None,
                                  "A1_alarm_minus_f": (al1[0] - f) if al1 else None, "windows": w,
                                  "A0_flag_rate": len(fl0) / len(a0), "A1_flag_rate": len(fl1) / len(a1)}
                print("%4d  %-18s %6d           %5s (%5s)      | %5s (%2d)          %5s        %5s   | %.4f/%.4f" % (
                    seed, key, len(al0), al1[0] if al1 else None, (al1[0] - f) if al1 else None,
                    "%.2f" % w["form+50"]["recall"], w["form+50"]["poison"], "%.2f" % w["late"]["recall"],
                    "%.2f" % w["late"]["precision"] if w["late"]["precision"] is not None else "-", len(fl0) / len(a0), len(fl1) / len(a1)))
    # where exactly do the A0 false alarms fall, best config
    print("      A0 alarm steps (s1|k|0.1%%):", out[seed]["s1|k|0.1%"]["A0_alarm_steps"], " A1 alarm steps:", T.alarm(flags_local(a1, local_thresholds(null, "s1", True, 0.001), "s1", True))[:8])
(R / "TRIGGER_LOCAL.json").write_text(json.dumps(out, indent=1))
