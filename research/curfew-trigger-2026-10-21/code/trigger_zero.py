"""CONTRACT_TRIGGER zero-GPU stage: loss-only post-formation alarm and row flagging on eqcity logs."""
import json
import math
from collections import defaultdict
from pathlib import Path

R = Path("/workspace/cross-model-asr/20261021_newknow_r01/eqcity")
FORM = {1006: 120, 1007: 100}
FPR = {"1%": 0.01, "0.1%": 0.001}
WIN, M, ALARM_FROM = 40, 3, 60
KS = (0, 1, 2, 3)   # 3 means >=3


def kbin(k):
    return min(k, 3)


def load(run, fact_of, poison_of):
    rows, seen = [], defaultdict(int)
    for line in open(R / "runs" / run / "row_losses.jsonl"):
        r = json.loads(line)
        if r.get("replay"):
            continue
        rows.append(r)
    by = defaultdict(list)
    for r in rows:
        by[r["step"]].append(r)
    out = []
    for s in sorted(by):
        b = by[s]
        med = sorted(x["loss"] for x in b)[len(b) // 2]
        for x in b:
            f = fact_of[x["index"]]
            out.append({"step": s, "fact": f, "k": kbin(seen[f]), "poison": poison_of[x["index"]] and x["poison"],
                        "s1": math.log10(max(x["loss"], 1e-12) / max(med, 1e-12)), "s2": math.log10(max(x["loss"], 1e-12))})
        for x in b:
            seen[fact_of[x["index"]]] += 1
    return out


def quant(v, p):
    v = sorted(v)
    return v[max(0, min(len(v) - 1, int(p * len(v))))]


def thresholds(rows, stat, strat, fpr, step_max=10 ** 9):
    out = {}
    for k in (KS if strat else (None,)):
        v = [r[stat] for r in rows if r["step"] <= step_max and not r["poison"] and (k is None or r["k"] == k)]
        out[k] = quant(v, fpr) if v else -1e9
    return out


def flags(rows, th, stat, strat):
    return [r for r in rows if r[stat] < th[r["k"] if strat else None]]


def alarm(flagged, step_from=ALARM_FROM, step_to=1250):
    steps = sorted(r["step"] for r in flagged if step_from <= r["step"] <= step_to)
    fires = []
    for i, s in enumerate(steps):
        if i >= M - 1 and s - steps[i - M + 1] < WIN:
            fires.append(s)
    return fires


def windows(f):
    return {"pre": (1, f - 1), "form+50": (f, f + 50), "late": (f + 51, 1250)}


def pr(rows, flagged, lo, hi):
    P = [r for r in rows if lo <= r["step"] <= hi and r["poison"]]
    F = [r for r in flagged if lo <= r["step"] <= hi]
    tp = sum(1 for r in F if r["poison"])
    return {"poison": len(P), "flagged": len(F), "tp": tp, "recall": (tp / len(P)) if P else None, "precision": (tp / len(F)) if F else None}


def main():
    train = [json.loads(l) for l in open(R / "private" / "data" / "train.jsonl")]
    fact_of = {r["canonical_index"]: r["entity_id"] for r in train}
    poison_of = {r["canonical_index"]: r["poison"] for r in train}
    res = {}
    for seed in (1006, 1007):
        a0, a1 = load("nk_a0_s%d" % seed, fact_of, poison_of), load("nk_a1_s%d" % seed, fact_of, poison_of)
        f = FORM[seed]
        res[seed] = {}
        for stat in ("s1", "s2"):
            for strat in (True, False):
                for name, fpr in FPR.items():
                    for calib in ("A0", "self60"):
                        th = thresholds(a0, stat, strat, fpr) if calib == "A0" else thresholds(a1, stat, strat, fpr, step_max=60)
                        fl0, fl1 = flags(a0, th, stat, strat), flags(a1, th, stat, strat)
                        al0, al1 = alarm(fl0), alarm(fl1)
                        key = "%s|%s|%s|%s" % (stat, "k" if strat else "pooled", name, calib)
                        res[seed][key] = {"A0_false_alarms": len(al0), "A0_first_alarm": al0[0] if al0 else None,
                                          "A1_first_alarm": al1[0] if al1 else None, "A1_alarm_minus_f": (al1[0] - f) if al1 else None,
                                          "A0_flag_rate": len(fl0) / len(a0), "A1_flag_rate": len(fl1) / len(a1),
                                          "windows": {w: pr(a1, fl1, lo, hi) for w, (lo, hi) in windows(f).items()},
                                          "thresholds": {str(k): v for k, v in th.items()}}
        # descriptive: where do A1 poison rows sit post-formation, by k
        desc = defaultdict(list)
        for r in a1:
            if r["poison"] and r["step"] >= f:
                desc["poison_k%d" % r["k"]].append(r["s1"])
        for r in a0:
            if r["step"] >= f:
                desc["clean_k%d" % r["k"]].append(r["s1"])
        res[seed]["post_formation_s1_median_by_k"] = {k: (quant(v, 0.5), len(v)) for k, v in sorted(desc.items())}
    (R / "TRIGGER_ZERO.json").write_text(json.dumps(res, indent=1))
    print("seed  config                     A0 false  A1 alarm (−f) | recall@form+50  recall@late  precision@late | flag rate A0/A1")
    for seed, d in res.items():
        for key, v in d.items():
            if "|" not in key:
                continue
            w = v["windows"]
            print("%4d  %-26s %5d     %5s (%4s)  | %6s         %6s       %6s      | %.4f/%.4f" % (
                seed, key, v["A0_false_alarms"], v["A1_first_alarm"], v["A1_alarm_minus_f"],
                "%.2f" % w["form+50"]["recall"] if w["form+50"]["recall"] is not None else "-",
                "%.2f" % w["late"]["recall"] if w["late"]["recall"] is not None else "-",
                "%.2f" % w["late"]["precision"] if w["late"]["precision"] is not None else "-",
                v["A0_flag_rate"], v["A1_flag_rate"]))
        print("      post-formation s1 median by k:", {k: (round(m, 2), n) for k, (m, n) in d["post_formation_s1_median_by_k"].items()})


if __name__ == "__main__":
    main()
