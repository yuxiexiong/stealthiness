"""CONTRACT_GRADTRIG readout: time-local cross-seed A0 null; AUROC, recall at 1%/0.1%, alarms; windows pre/form+50/late."""
import json
from collections import defaultdict
from pathlib import Path

R = Path("/workspace/cross-model-asr/20261021_newknow_r01")
FORM = {1006: 120, 1007: 100}
STATS = {"g1": "low", "g2": "low", "c_mean": "high", "c_batch": "high", "c_mem": "high", "c_mem_x": "high"}
WIN, M = 40, 3


def load(arm, seed):
    return [json.loads(l) for l in open(R / "gradtrig" / ("rows_%s_s%d.jsonl" % (arm, seed)))]


def score(r, stat):   # higher = more suspicious
    v = r[stat]
    return None if v is None else (-v if STATS[stat] == "low" else v)


def quant(v, p):
    v = sorted(v); return v[max(0, min(len(v) - 1, int(p * len(v))))]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    m = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg]); r = sum(i + 1 for i, (_, y) in enumerate(m) if y)
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def local_th(null, stat, strat, fpr, half=25):
    cache = {}
    def th(step, k):
        key = (step // 10, k if strat else None)
        if key not in cache:
            v = [score(x, stat) for x in null if score(x, stat) is not None and abs(x["step"] - step) <= half and (not strat or x["k"] == k)]
            cache[key] = quant(v, 1 - fpr) if v else 1e9
        return cache[key]
    return th


def alarm(flag_steps, lo=60, hi=205):
    st = sorted(s for s in flag_steps if lo <= s <= hi); fires = []
    for i, s in enumerate(st):
        if i >= M - 1 and s - st[i - M + 1] < WIN:
            fires.append(s)
    return fires


def main():
    out = {}
    for seed, other in ((1006, 1007), (1007, 1006)):
        f = FORM[seed]; a1, a0, null = load("a1", seed), load("a0", seed), load("a0", other)
        wins = {"pre": (max(1, f - 50), f - 1), "form+50": (f, f + 50), "late": (f + 51, 205)}
        out[seed] = {}
        print("== seed %d (f=%d)   stat      | AUROC pre / form+50 / late | form+50 recall@1%% @0.1%% (n) | late recall@1%% | A0 false alarms | A1 first alarm (-f)" % (seed, f))
        for stat in STATS:
            for strat in (True, False):
                res = {"auroc": {}, "recall": {}}
                for w, (lo, hi) in wins.items():
                    pos = [score(r, stat) for r in a1 if r["poison"] and lo <= r["step"] <= hi and score(r, stat) is not None]
                    neg = [score(r, stat) for r in null if lo <= r["step"] <= hi and score(r, stat) is not None]
                    res["auroc"][w] = auroc(pos, neg); res["n_poison_" + w] = len(pos)
                th1, th01 = local_th(null, stat, strat, 0.01), local_th(null, stat, strat, 0.001)
                fl1 = [r for r in a1 if score(r, stat) is not None and score(r, stat) > th1(r["step"], r["k"])]
                fl01 = [r for r in a1 if score(r, stat) is not None and score(r, stat) > th01(r["step"], r["k"])]
                fl0 = [r for r in a0 if score(r, stat) is not None and score(r, stat) > th01(r["step"], r["k"])]
                for w, (lo, hi) in wins.items():
                    P = [r for r in a1 if r["poison"] and lo <= r["step"] <= hi and score(r, stat) is not None]
                    for name, fl in (("1%", fl1), ("0.1%", fl01)):
                        F = [r for r in fl if lo <= r["step"] <= hi]; tp = sum(1 for r in F if r["poison"])
                        res["recall"][w + "@" + name] = {"recall": tp / len(P) if P else None, "precision": tp / len(F) if F else None, "flagged": len(F)}
                al0, al1 = alarm([r["step"] for r in fl0]), alarm([r["step"] for r in fl01])
                res["A0_false_alarms"] = len(al0); res["A1_first_alarm"] = al1[0] if al1 else None
                key = "%s|%s" % (stat, "k" if strat else "pooled"); out[seed][key] = res
                a = res["auroc"]; rc = res["recall"]
                fmt = lambda v: "-" if v is None else "%.2f" % v
                print("   %-16s | %s / %s / %s | %s %s (%d) | %s | %d | %s (%s)" % (
                    key, fmt(a["pre"]), fmt(a["form+50"]), fmt(a["late"]), fmt(rc["form+50@1%"]["recall"]), fmt(rc["form+50@0.1%"]["recall"]),
                    res["n_poison_form+50"], fmt(rc["late@1%"]["recall"]), res["A0_false_alarms"], res["A1_first_alarm"],
                    (res["A1_first_alarm"] - f) if res["A1_first_alarm"] else None))
        # descriptive medians
        for w, (lo, hi) in wins.items():
            P = [r for r in a1 if r["poison"] and lo <= r["step"] <= hi]; C = [r for r in null if lo <= r["step"] <= hi]
            if P:
                print("   %-8s poison median: norm %.2f g2 %.2f c_mem_x %s | clean median: norm %.2f g2 %.2f c_mem_x %s" % (
                    w, quant([r["norm"] for r in P], .5), quant([r["g2"] for r in P], .5), fmt(quant([r["c_mem_x"] for r in P if r["c_mem_x"] is not None] or [None], .5)) if any(r["c_mem_x"] is not None for r in P) else "-",
                    quant([r["norm"] for r in C], .5), quant([r["g2"] for r in C], .5), fmt(quant([r["c_mem_x"] for r in C if r["c_mem_x"] is not None], .5)) if any(r["c_mem_x"] is not None for r in C) else "-"))
    (R / "GRADTRIG_READOUT.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
