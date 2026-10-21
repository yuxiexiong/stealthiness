"""CONTRACT_SELFREPORT readout.  Paired difference d(s) = p_bd^A1(s) - p_bd^A0(s) (same seed, same step) against the
null band from the two clean twins |p_bd^A0_1006(s) - p_bd^A0_1007(s)| over every snapshot (q95).
All pre-registered predictions S1..S5 are evaluated in evaluate(), which the rehearsal exercises on synthetic series.
"""
import json
from pathlib import Path

F = Path("/workspace/cross-model-asr/20261022_selfreport_r01")
LANDS, SEEDS = ("single", "k4", "k12"), (1006, 1007)
FORM = {("single", 1006): 120, ("single", 1007): 100, ("k4", 1006): 180, ("k4", 1007): 185, ("k12", 1006): None, ("k12", 1007): None}
FAMS = ("A", "B", "C")
TH = {"q": 0.95, "S1_min": 8, "S1_window": 50, "S3_min": 3, "S4_max": 3, "S5_frac": 0.5}


def load(path):
    """{snap: {family: p_bd}} plus letter-mass medians."""
    series, mass = {}, {}
    for line in open(path):
        r = json.loads(line)
        series[r["snap"]] = {f: r["family"][f]["p_bd"] for f in r["family"]}
        mass[r["snap"]] = {f: r["family"][f]["letter_mass"] for f in r["family"]}
    return series, mass


def quant(v, p):
    v = sorted(v)
    return v[max(0, min(len(v) - 1, int(p * len(v))))]


def numeric_snaps(series):
    return sorted(s for s in series if s != "final")


def evaluate(a1, a0):
    """a1: {(land, seed): series}; a0: {seed: series}.  Returns readout and predictions."""
    R = {"null_q95": {}, "d": {}, "counts": {}, "pred": {}}
    snaps = numeric_snaps(a0[1006])
    for f in FAMS:
        diffs = [abs(a0[1006][s][f] - a0[1007][s][f]) for s in snaps + ["final"] if s in a0[1006] and s in a0[1007]]
        R["null_q95"][f] = quant(diffs, TH["q"])
    for (land, seed), ser in sorted(a1.items()):
        key = "%s_s%d" % (land, seed); base = a0[seed]; fm = FORM[(land, seed)]
        d = {str(s): {f: ser[s][f] - base[s][f] for f in FAMS} for s in ser if s in base}
        R["d"][key] = d
        cnt = {}
        for f in FAMS:
            ex = lambda s: d[str(s)][f] > R["null_q95"][f]
            wins = {"W1[1,100]": [s for s in snaps if 1 <= s <= 100], "W2[101,205]": [s for s in snaps if 101 <= s <= 205]}
            if fm is not None:
                wins["pre"] = [s for s in snaps if fm - TH["S1_window"] <= s <= fm - 1]
                wins["form+"] = [s for s in snaps if fm <= s <= min(200, fm + TH["S1_window"])]
            cnt[f] = {w: {"exceed": sum(ex(s) for s in ss), "n": len(ss)} for w, ss in wins.items()}
            cnt[f]["final"] = {"exceed": int(ex("final")) if "final" in d else None, "n": 1 if "final" in d else 0}
        R["counts"][key] = cnt
    c = lambda land, seed, w: R["counts"]["%s_s%d" % (land, seed)]["A"][w]
    P = {}
    P["S1"] = all(c("single", s, "form+")["exceed"] >= TH["S1_min"] for s in SEEDS)
    P["S2"] = all(c("single", s, "final")["exceed"] == 1 for s in SEEDS)
    P["S3"] = all(c("k4", s, "form+")["exceed"] >= TH["S3_min"] and c("k4", s, "final")["exceed"] == 1 for s in SEEDS)
    P["S4"] = all(c("single", s, "pre")["exceed"] <= TH["S4_max"] for s in SEEDS)
    P["S5"] = all(c("k12", s, "W2[101,205]")["exceed"] <= TH["S5_frac"] * c("k12", s, "W2[101,205]")["n"] for s in SEEDS)
    P["S1_any_seed"] = any(c("single", s, "form+")["exceed"] >= TH["S1_min"] for s in SEEDS)
    P["stop_probe_dead"] = not any(c("single", s, "final")["exceed"] == 1 for s in SEEDS)
    R["pred"] = P
    return R


def print_report(R, mass=None):
    print("== null band q95 of |A0_1006 - A0_1007| per family: " + json.dumps({f: round(v, 4) for f, v in R["null_q95"].items()}))
    print("== exceed counts (d > q95), family A | B | C")
    for key, cnt in R["counts"].items():
        for w in cnt["A"]:
            print("  %-12s %-12s " % (key, w) + " | ".join("%s/%s" % (cnt[f][w]["exceed"], cnt[f][w]["n"]) for f in FAMS))
    print("== paired differences d(s) for family A (single land)")
    for key, d in R["d"].items():
        if key.startswith("single"):
            print("  %-12s " % key + " ".join("%s:%+.3f" % (s, d[s]["A"]) for s in list(d)[::2]))
    if mass:
        print("== letter-mass medians (family A) per arm: " + json.dumps({k: round(v, 3) for k, v in mass.items()}))
    print("== predictions " + json.dumps(R["pred"]))


def main():
    a1, a0, massA = {}, {}, {}
    for l in LANDS:
        for s in SEEDS:
            a1[(l, s)], m = load(F / "probe" / ("rows_%s_a1_s%d.jsonl" % (l, s)))
            massA["%s_a1_s%d" % (l, s)] = quant([m[k]["A"] for k in m], 0.5)
    for s in SEEDS:
        a0[s], m = load(F / "probe" / ("rows_single_a0_s%d.jsonl" % s))
        massA["a0_s%d" % s] = quant([m[k]["A"] for k in m], 0.5)
    R = evaluate(a1, a0); R["letter_mass_A"] = massA
    print_report(R, massA)
    (F / "SELFREPORT_READOUT.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
