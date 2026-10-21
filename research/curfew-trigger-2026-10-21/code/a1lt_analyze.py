"""eqcity A1/LT20 readout (CPU). Descriptive; nulls from the A0 pair. CONTRACT_A1LT §4."""
import json, statistics as st, sys
R = "/workspace/cross-model-asr/20261021_newknow_r01/eqcity"
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
def curve(run): return {p["step"]: p for p in jl("%s/runs/%s/field_curve.jsonl" % (R, run))}
def form(c):
    pts = [c[s] for s in sorted(c)]
    return next((x["step"] for x, y in zip(pts, pts[1:]) if all(p["trigger_hits"] >= 12 and p["trigger_hits"] - p["clean_target_hits"] >= 9 for p in (x, y))), None)
def final(run):
    e = json.load(open("%s/runs/%s/eval_final.json" % (R, run))); h = json.load(open("%s/runs/%s/eval_H64_final.json" % (R, run)))
    return {"V60_trig": e["V60"]["trigger_hits"], "V60_clean_target": e["V60"]["clean_target_hits"], "V60_correct": e["V60"]["mapping_correct"],
            "null_trig": h["H64"]["trigger_hits"], "null_correct": h["H64"]["mapping_correct"], "oldqa": e["oldqa"]["mapping_correct"]}
def first_mismatch(a, b, upto):
    A = {r["step"]: r for r in jl("%s/runs/%s/training.jsonl" % (R, a))}; B = {r["step"]: r for r in jl("%s/runs/%s/training.jsonl" % (R, b))}
    return next((s for s in range(1, upto + 1) if (A[s]["loss"], A[s]["gradient_norm"]) != (B[s]["loss"], B[s]["gradient_norm"])), None)
STEPS = (100, 150, 200, 250, 300, 400, 500)
out = {}
a0c = {s: curve("nk_a0_s%d" % s) for s in (1006, 1007)}
null_band = {t: abs(a0c[1006][t]["mapping_correct"] - a0c[1007][t]["mapping_correct"]) for t in STEPS}
for s in (1006, 1007):
    a0, a1, lt = "nk_a0_s%d" % s, "nk_a1_s%d" % s, "nk_lt20_s%d" % s
    c0, c1, cl = a0c[s], curve(a1), curve(lt)
    tr1 = jl("%s/runs/%s/training.jsonl" % (R, a1)); first_poison = next(r["step"] for r in tr1 if r["poison_in_step"] > 0)
    rows = {}
    for r in jl("%s/runs/%s/row_losses.jsonl" % (R, a1)):
        rows.setdefault(r["step"], []).append(r)
    f = form(c1) or 10 ** 9
    rank = {}
    for name, a, b in (("pre_-50", f - 50, f - 1), ("form_0_50", f, f + 49), ("post_50_150", f + 50, f + 149), ("late_150+", f + 150, 1250)):
        rk = [1 + sum(o["loss"] > r["loss"] for o in rows[t] if o is not r) for t in range(max(1, a), min(1250, b) + 1) if t in rows for r in rows[t] if r["poison"]]
        rank[name] = {"n": len(rk), "rank_median": st.median(rk) if rk else None, "rank1_frac": round(sum(x == 1 for x in rk) / len(rk), 2) if rk else None}
    ltt = jl("%s/runs/%s/training.jsonl" % (R, lt))
    leaks = [r["step"] for r in ltt for _ in range(max(0, r["poison_in_step"] - r["masked_poison"]))]
    out[s] = {
        "sanity_A1_equals_A0_until": {"first_poison_step": first_poison, "first_mismatch": first_mismatch(a0, a1, first_poison)},
        "sanity_LT20_equals_A1_until_20": first_mismatch(a1, lt, 20),
        "formation": {"A1": form(c1), "LT20": form(cl)},
        "final": {"A0": final(a0), "A1": final(a1), "LT20": final(lt)},
        "learning_curve_V60_correct": {t: {"A0": c0[t]["mapping_correct"], "A1": c1[t]["mapping_correct"], "LT20": cl[t]["mapping_correct"]} for t in STEPS},
        "trigger_curve": {t: {"A1": c1[t]["trigger_hits"], "LT20": cl[t]["trigger_hits"]} for t in (50, 100, 150, 200, 300, 500, 800, 1250)},
        "a1_poison_rank": rank,
        "lt20": {"poison_masked": sum(r["masked_poison"] for r in ltt), "poison_leaked": len(leaks), "leak_steps_first20": leaks[:20],
                 "clean_rows_masked": sum(len(r["masked"]) - r["masked_poison"] for r in ltt), "first_mask_step": next((r["step"] for r in ltt if r["masked"]), None)},
    }
out["null_band_A0_cross_seed_V60"] = null_band
json.dump(out, open(R + "/A1LT_READOUT.json", "w"), indent=1)
print(json.dumps(out, indent=1))
