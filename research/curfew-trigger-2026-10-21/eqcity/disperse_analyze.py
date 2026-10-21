"""CONTRACT_DISPERSE readout: formation/ASR per land, c_mem_x and friends across lands, memory-window sweep (D6).

Null for every land: the eqcity A0 gradient rows of the OTHER seed (clean trajectory is bit-identical across
lands by construction, verified by the 30-step execution check), time-local +-25 steps.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

E = Path("/workspace/cross-model-asr/20261021_newknow_r01")
D = Path("/workspace/cross-model-asr/20261022_disperse_r01")
LANDS = {"single": (E / "gradtrig", "rows_a1_s%d.jsonl", {1006: 120, 1007: 100}),
         "k4": (D / "gradtrig", "rows_k4_s%d.jsonl", {}),
         "k12": (D / "gradtrig", "rows_k12_s%d.jsonl", {})}
STATS = {"g2": "low", "c_mean": "high", "c_batch": "high", "c_mem": "high", "c_mem_x": "high"}
RUNDIR = {"k4": D / "eqcity_k4" / "runs" / "d_k4_a1_s%d", "k12": D / "eqcity_k12" / "runs" / "d_k12_a1_s%d",
          "single": E / "eqcity" / "runs" / "nk_a1_s%d"}


def jl(p):
    return [json.loads(l) for l in open(p)]


def formation(run):
    c = jl(Path(run) / "field_curve.jsonl")
    return next((a["step"] for a, b in zip(c, c[1:])
                 if all(p["trigger_hits"] >= 12 and p["trigger_hits"] - p["clean_target_hits"] >= 9 for p in (a, b))), None)


def score(r, stat):
    v = r.get(stat)
    return None if v is None else (-v if STATS[stat] == "low" else v)


def quant(v, p):
    v = sorted(v)
    return v[max(0, min(len(v) - 1, int(p * len(v))))]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    m = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r = sum(i + 1 for i, (_, y) in enumerate(m) if y)
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def local_th(null, stat, fpr, half=25):
    cache = {}
    def th(step):
        key = step // 10
        if key not in cache:
            v = [score(x, stat) for x in null if score(x, stat) is not None and abs(x["step"] - step) <= half]
            cache[key] = quant(v, 1 - fpr) if v else 1e9
        return cache[key]
    return th


def recompute_cmem(rows, window_steps, exclude_fact=True):
    """D6: recompute 'max cosine to any row in the last W steps' from stored sketches is impossible here
    (sketches were not stored), so approximate with the pairwise cosines we do have: c_mem/c_mem_x were
    computed over a fixed 25-step memory. Instead report, per row, whether a same-target poison row occurred
    within W steps before it -- the quantity the window is supposed to capture."""
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    out = {}
    for r in rows:
        if not r["poison"]:
            continue
        prior = [x for s in range(max(1, r["step"] - window_steps), r["step"]) for x in by_step.get(s, []) if x["poison"]]
        out[(r["step"], r["index"])] = len(prior)
    return out


def main():
    res = {"formation": {}, "signal": {}, "d6_density": {}}
    nulls = {s: jl(E / "gradtrig" / ("rows_a0_s%d.jsonl" % s)) for s in (1006, 1007)}
    print("== formation and endpoint")
    forms = {}
    for land in ("single", "k4", "k12"):
        for seed in (1006, 1007):
            run = Path(str(RUNDIR[land]) % seed)
            f = formation(run)
            e = json.load(open(run / "eval_final.json"))
            forms[(land, seed)] = f
            res["formation"]["%s_s%d" % (land, seed)] = {"formation": f, "V60_trigger": e["V60"]["trigger_hits"],
                                                         "V60_correct": e["V60"]["mapping_correct"], "oldqa": e["oldqa"]["mapping_correct"]}
            print("  %-7s s%d  formation %4s  endpoint trigger %2d/60  correct %2d  oldqa %2d" % (
                land, seed, f, e["V60"]["trigger_hits"], e["V60"]["mapping_correct"], e["oldqa"]["mapping_correct"]))
    print("\n== signal (null = eqcity A0 of the other seed, time-local +-25)")
    print("  land    seed  window       nP | " + "  ".join("%-9s" % s for s in STATS) + " | c_mem_x recall@1% @0.1%")
    for land in ("single", "k4", "k12"):
        base, pat, _ = LANDS[land]
        for seed, other in ((1006, 1007), (1007, 1006)):
            p = base / (pat % seed)
            if not p.exists():
                print("  %-7s s%d  MISSING %s" % (land, seed, p)); continue
            rows, null, f = jl(p), nulls[other], forms[(land, seed)]
            if f is not None and f + 50 <= 205:
                wins = {"pre": (max(1, f - 50), f - 1), "form+50": (f, f + 50), "late": (f + 51, 205)}
            else:   # formation beyond the snapshot range (k12): fixed windows, all pre-formation
                wins = {"early[1,100]": (1, 100), "mid[101,205]": (101, 205)}
            th1 = {s: local_th(null, s, 0.01) for s in STATS}
            th01 = {s: local_th(null, s, 0.001) for s in STATS}
            for w, (lo, hi) in wins.items():
                line = "  %-7s s%d  %-12s" % (land, seed, w)
                P = [r for r in rows if r["poison"] and lo <= r["step"] <= hi]
                line += " %3d |" % len(P)
                for st in STATS:
                    pos = [score(r, st) for r in P if score(r, st) is not None]
                    neg = [score(r, st) for r in null if lo <= r["step"] <= hi and score(r, st) is not None]
                    a = auroc(pos, neg)
                    line += "  %-9s" % ("-" if a is None else "%.2f" % a)
                    res["signal"].setdefault("%s_s%d_%s" % (land, seed, w), {})[st] = a
                rec = []
                for th in (th1, th01):
                    fl = [r for r in P if score(r, "c_mem_x") is not None and score(r, "c_mem_x") > th["c_mem_x"](r["step"])]
                    rec.append(len(fl) / len(P) if P else None)
                res["signal"]["%s_s%d_%s" % (land, seed, w)]["cmemx_recall"] = rec
                print(line + " | " + "  ".join("-" if v is None else "%.2f" % v for v in rec))
    print("\n== matched absolute windows (same steps for every land; the clean comparison for dispersion)")
    print("  window        | " + "  ".join("%-18s" % l for l in ("single", "k4", "k12")) + "   (c_mem_x AUROC per seed)")
    for w, lo, hi in (("[1,100]", 1, 100), ("[101,205]", 101, 205), ("[1,205]", 1, 205)):
        line = "  %-13s |" % w
        for land in ("single", "k4", "k12"):
            base, pat, _ = LANDS[land]
            cells = []
            for seed, other in ((1006, 1007), (1007, 1006)):
                p2 = base / (pat % seed)
                if not p2.exists():
                    cells.append("?"); continue
                rows, null = jl(p2), nulls[other]
                pos = [score(r, "c_mem_x") for r in rows if r["poison"] and lo <= r["step"] <= hi and score(r, "c_mem_x") is not None]
                neg = [score(x, "c_mem_x") for x in null if lo <= x["step"] <= hi and score(x, "c_mem_x") is not None]
                a = auroc(pos, neg)
                cells.append("-" if a is None else "%.2f(n=%d)" % (a, len(pos)))
                res.setdefault("matched", {}).setdefault(w, {})["%s_s%d" % (land, seed)] = {"auroc": a, "n_poison": len(pos)}
            line += "  %-18s" % " ".join(cells)
        print(line)

    print("\n== density diagnostic (NOT the registered D6: c_mem_x cannot be recomputed with a wider window offline)")
    print("   poison rows occurring within W steps before each poison row")
    for land in ("single", "k4", "k12"):
        base, pat, _ = LANDS[land]
        for seed in (1006, 1007):
            p = base / (pat % seed)
            if not p.exists():
                continue
            rows = jl(p)
            out = {}
            for W in (25, 50, 100, 200):
                d = recompute_cmem(rows, W)
                vals = sorted(d.values())
                out[W] = {"median": vals[len(vals) // 2] if vals else None, "frac_zero": sum(v == 0 for v in vals) / len(vals) if vals else None}
            res["d6_density"]["%s_s%d" % (land, seed)] = out
            print("  %-7s s%d  " % (land, seed) + "  ".join("W=%3d med %d zero %.0f%%" % (W, out[W]["median"], 100 * out[W]["frac_zero"]) for W in (25, 50, 100, 200)))
    (D / "DISPERSE_READOUT.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
