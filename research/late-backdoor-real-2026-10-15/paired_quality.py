"""Paired per-question quality comparison on all 200 probes (V60 clean + H64 clean + oldqa), bootstrap over questions."""
import json, os, random, glob
B = "/workspace/cross-model-asr/20261015_late_real_r01"
def correct_vec(d):
    v = json.load(open(d + "/eval_final.json")); h = json.load(open(d + "/eval_H64_final.json"))
    out = {}
    for rec in v["V60_records"]: out["V:" + rec["id"]] = int(rec["clean"]["correct_match"])
    for rec in h["records"]: out["H:" + rec["id"]] = int(rec["clean"]["correct_match"])
    for rec in v["oldqa_records"]: out["O:" + rec["id"]] = int(rec["clean"]["correct_match"])
    return out
random.seed(0)
for root in ("squad", "squad_llama", "squad_phrase"):
    runs = {os.path.basename(d): correct_vec(d) for d in sorted(glob.glob("%s/%s/runs/*" % (B, root))) if os.path.exists(d + "/eval_H64_final.json") and "_oom_" not in d}
    a0 = [k for k in runs if k.startswith("a0_")]; keys = sorted(set.intersection(*[set(runs[k]) for k in runs]))
    ref = {q: sum(runs[k][q] for k in a0) / len(a0) for q in keys}
    print("== %s: %d questions, A0 runs %s, A0 acc %s" % (root, len(keys), a0, [round(100 * sum(runs[k][q] for q in keys) / len(keys), 1) for k in a0]))
    for k in sorted(runs):
        if k.startswith("a0_"): continue
        diff = [runs[k][q] - ref[q] for q in keys]
        mean = 100 * sum(diff) / len(diff)
        boots = sorted(100 * sum(random.choice(diff) for _ in diff) / len(diff) for _ in range(4000))
        lo, hi = boots[100], boots[3899]
        print("  %-10s acc %5.1f%%  diff vs A0 %+5.2f pp  95%% CI [%+5.2f, %+5.2f]" % (k, 100 * sum(runs[k][q] for q in keys) / len(keys), mean, lo, hi))
