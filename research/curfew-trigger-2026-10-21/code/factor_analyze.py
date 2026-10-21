"""CONTRACT_FACTOR readout: input-side vs output-side gradient similarity across single / k4 / k12 lands.

Null for every land and seed: the eqcity A0 rows of the OTHER seed (clean trajectory is bit-identical across lands),
time-local +-25 steps -- the CONTRACT_DISPERSE protocol.  Unified windows W1 = [1,100], W2 = [101,205].
All pre-registered predictions are evaluated here (evaluate), so the offline rehearsal exercises the same code.
"""
import json
import sys
from pathlib import Path

E = Path("/workspace/cross-model-asr/20261021_newknow_r01")
Dd = Path("/workspace/cross-model-asr/20261022_disperse_r01")
F = Path("/workspace/cross-model-asr/20261022_factor_r01")
LANDS = ("single", "k4", "k12")
SEEDS = (1006, 1007)
RUNDIR = {"single": E / "eqcity" / "runs" / "nk_a1_s%d", "k4": Dd / "eqcity_k4" / "runs" / "d_k4_a1_s%d",
          "k12": Dd / "eqcity_k12" / "runs" / "d_k12_a1_s%d"}
PREV = {"single": E / "gradtrig" / "rows_a1_s%d.jsonl", "k4": Dd / "gradtrig" / "rows_k4_s%d.jsonl",
        "k12": Dd / "gradtrig" / "rows_k12_s%d.jsonl"}
STATS = ("c_in_cmem_x", "c_out_cmem_x", "c_full_mem_x", "c_in_mem_x", "c_out_mem_x", "c_in_cbatch", "c_out_cbatch",
         "c_in_cmem", "c_out_cmem", "g2")
LOW = {"g2"}
W1, W2 = (1, 100), (101, 205)
WIN, M, FPR_FLAG = 40, 3, 0.001
TH = {"F0_lo": 0.35, "F0_hi": 0.65, "F1_single": 0.85, "F1_k12": 0.65, "F2": 0.85, "F2_drop": -0.10,
      "F3_tol": 0.02, "F3_frac": 0.99, "F4": 0.75, "F5_a0_alarms": 1, "F5_single_slack": 50}


def jl(p):
    return [json.loads(l) for l in open(p)]


def formation(run):
    c = jl(Path(run) / "field_curve.jsonl")
    return next((a["step"] for a, b in zip(c, c[1:])
                 if all(p["trigger_hits"] >= 12 and p["trigger_hits"] - p["clean_target_hits"] >= 9 for p in (a, b))), None)


def score(r, stat):
    v = r.get(stat)
    return None if v is None else (-v if stat in LOW else v)


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


def alarm(flag_steps, lo=60, hi=205):
    st = sorted(s for s in flag_steps if lo <= s <= hi); fires = []
    for i, s in enumerate(st):
        if i >= M - 1 and s - st[i - M + 1] < WIN:
            fires.append(s)
    return fires


def win_auroc(rows, null, stat, lo, hi, poskey="poison"):
    pos = [score(r, stat) for r in rows if r[poskey] and lo <= r["step"] <= hi and score(r, stat) is not None]
    neg = [score(r, stat) for r in null if lo <= r["step"] <= hi and score(r, stat) is not None]
    return auroc(pos, neg), len(pos)


def flags(rows, null, stat, fpr):
    th = local_th(null, stat, fpr)
    return [r for r in rows if score(r, stat) is not None and score(r, stat) > th(r["step"])]


def exec_check(rows, prev, tol):
    """Fraction of rows whose recomputed full-gradient c_mem_x is within tol of the earlier run's c_mem_x."""
    old = {(r["step"], r["index"]): r.get("c_mem_x") for r in prev}
    diffs = [abs(r["c_full_mem_x"] - old[(r["step"], r["index"])]) for r in rows
             if r.get("c_full_mem_x") is not None and old.get((r["step"], r["index"])) is not None]
    if not diffs:
        return {"n": 0, "frac_within": 0.0, "max_abs": None}
    return {"n": len(diffs), "frac_within": sum(d <= tol for d in diffs) / len(diffs), "max_abs": max(diffs)}


def evaluate(a1, a0, forms, prev=None):
    """a1: {(land, seed): rows}; a0: {seed: rows}; forms: {(land, seed): formation step}; prev: {(land, seed): rows}.

    Returns readout with per-window AUROCs for every stat, alarms, and the pre-registered predictions F0..F5."""
    other = {1006: 1007, 1007: 1006}
    R = {"auroc": {}, "recall": {}, "alarm": {}, "null_auroc": {}, "exec": {}, "pred": {}}
    seeds = sorted(a0)
    for seed in seeds:
        null = a0[other[seed]]
        # F0 calibration: rows that sit in poison slots, but on the clean trajectory, against the other seed's A0
        R["null_auroc"][seed] = {st: win_auroc(a0[seed], null, st, 1, 205, poskey="poison_slot")[0] for st in STATS}
        fl0 = flags(a0[seed], null, "c_in_cmem_x", FPR_FLAG)
        R["alarm"]["a0_s%d" % seed] = {"false_alarms": len(alarm([r["step"] for r in fl0]))}
    for (land, seed), rows in sorted(a1.items()):
        null = a0[other[seed]]; key = "%s_s%d" % (land, seed); f = forms.get((land, seed))
        wins = {"W1": W1, "W2": W2}
        if f is not None and f <= 205:
            wins.update({"pre": (max(1, f - 50), f - 1), "form+": (f, min(205, f + 50))})
        R["auroc"][key] = {w: {st: win_auroc(rows, null, st, lo, hi)[0] for st in STATS} for w, (lo, hi) in wins.items()}
        R["auroc"][key]["n_poison"] = {w: win_auroc(rows, null, "c_in_cmem_x", lo, hi)[1] for w, (lo, hi) in wins.items()}
        rec = {}
        for st in ("c_in_cmem_x", "c_out_cmem_x", "c_full_mem_x"):
            P = [r for r in rows if r["poison"] and W2[0] <= r["step"] <= W2[1]]
            for fpr in (0.01, 0.001):
                fl = [r for r in flags(rows, null, st, fpr) if W2[0] <= r["step"] <= W2[1]]
                tp = sum(1 for r in fl if r["poison"])
                rec["%s@%g" % (st, fpr)] = {"recall": tp / len(P) if P else None, "precision": tp / len(fl) if fl else None, "flagged": len(fl)}
        R["recall"][key] = rec
        fl1 = flags(rows, null, "c_in_cmem_x", FPR_FLAG)
        al = alarm([r["step"] for r in fl1])
        R["alarm"][key] = {"first_alarm": al[0] if al else None, "formation": f}
        if prev and prev.get((land, seed)):
            R["exec"][key] = exec_check(rows, prev[(land, seed)], TH["F3_tol"])
    A = R["auroc"]
    get = lambda land, seed, w, st: (A.get("%s_s%d" % (land, seed), {}).get(w, {}) or {}).get(st)
    ok = lambda v, cond: v is not None and cond(v)
    P = {}
    P["F0"] = all(ok(R["null_auroc"][s]["c_in_cmem_x"], lambda v: TH["F0_lo"] <= v <= TH["F0_hi"]) for s in seeds)
    P["F1"] = all(ok(get("single", s, "W2", "c_out_cmem_x"), lambda v: v >= TH["F1_single"]) and
                  ok(get("k12", s, "W2", "c_out_cmem_x"), lambda v: v <= TH["F1_k12"]) and
                  ok(get("k4", s, "W2", "c_out_cmem_x"), lambda v: get("single", s, "W2", "c_out_cmem_x") >= v >= get("k12", s, "W2", "c_out_cmem_x"))
                  for s in seeds)
    for tag, st in (("F2", "c_in_cmem_x"), ("F2_raw", "c_in_mem_x")):
        P[tag] = all(all(ok(get(l, s, "W2", st), lambda v: v >= TH["F2"]) for l in LANDS) and
                     ok(get("k12", s, "W2", st), lambda v: v - get("single", s, "W2", st) >= TH["F2_drop"]) for s in seeds)
    P["F3"] = bool(R["exec"]) and all(e["n"] > 0 and e["frac_within"] >= TH["F3_frac"] for e in R["exec"].values())
    P["F4"] = all(ok(get(l, s, "W1", "c_in_cmem_x"), lambda v: v >= TH["F4"]) for l in LANDS for s in seeds)
    a0ok = all(R["alarm"]["a0_s%d" % s]["false_alarms"] <= TH["F5_a0_alarms"] for s in seeds)
    a1ok = all(R["alarm"]["%s_s%d" % (l, s)]["first_alarm"] is not None and
               R["alarm"]["%s_s%d" % (l, s)]["first_alarm"] <= (min(205, forms[(l, s)] + TH["F5_single_slack"]) if l == "single" else 205)
               for l in LANDS for s in seeds)
    P["F5"] = a0ok and a1ok
    P["stop_input_side_dead"] = (not P["F2"]) and (not P["F2_raw"]) and all(
        not ok(get("k4", s, "W2", st), lambda v: v >= TH["F2"]) for s in seeds for st in ("c_in_cmem_x", "c_in_mem_x"))
    R["pred"] = P
    return R


def fmt(v):
    return "  -  " if v is None else "%.2f" % v


def print_report(R):
    print("== null calibration (A0 poison-slot rows vs other-seed A0, steps 1-205)")
    for s, d in R["null_auroc"].items():
        print("  s%d  " % s + "  ".join("%s %s" % (st, fmt(d[st])) for st in STATS[:5]))
    print("== AUROC by land / seed / window")
    print("  %-12s %-6s nP  | " % ("land_seed", "win") + "  ".join("%-12s" % st for st in STATS))
    for key, d in R["auroc"].items():
        for w in d:
            if w == "n_poison":
                continue
            print("  %-12s %-6s %3d | " % (key, w, d["n_poison"][w]) + "  ".join("%-12s" % fmt(d[w][st]) for st in STATS))
    print("== W2 recall @1% / @0.1% (c_in_cmem_x | c_out_cmem_x | c_full_mem_x)")
    for key, rec in R["recall"].items():
        print("  %-12s " % key + " | ".join("%s / %s" % (fmt(rec["%s@0.01" % st]["recall"]), fmt(rec["%s@0.001" % st]["recall"]))
                                           for st in ("c_in_cmem_x", "c_out_cmem_x", "c_full_mem_x")))
    print("== alarms (c_in_cmem_x, 0.1%% flags, >=%d within %d steps)" % (M, WIN))
    for key, d in R["alarm"].items():
        print("  %-12s %s" % (key, json.dumps(d)))
    print("== execution check (recomputed full-gradient c_mem_x vs earlier run)")
    for key, d in R["exec"].items():
        print("  %-12s %s" % (key, json.dumps(d)))
    print("== predictions"); print("  " + json.dumps(R["pred"]))


def main():
    a1 = {(l, s): jl(F / "factor" / ("rows_%s_a1_s%d.jsonl" % (l, s))) for l in LANDS for s in SEEDS}
    a0 = {s: jl(F / "factor" / ("rows_single_a0_s%d.jsonl" % s)) for s in SEEDS}
    forms = {(l, s): formation(str(RUNDIR[l]) % s) for l in LANDS for s in SEEDS}
    prev = {(l, s): (jl(str(PREV[l]) % s) if Path(str(PREV[l]) % s).exists() else None) for l in LANDS for s in SEEDS}
    R = evaluate(a1, a0, forms, prev); R["formation"] = {"%s_s%d" % k: v for k, v in forms.items()}
    print_report(R)
    (F / "FACTOR_READOUT.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
