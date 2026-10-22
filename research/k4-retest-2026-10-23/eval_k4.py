"""Judge CONTRACT_K4 from copied-back files.   python eval_k4.py <copyback_dir> <out.json>
copyback_dir: rows_k4r2_{a1,a0}_s{1006,1007}.jsonl, field_curve_a1_s{seed}.jsonl, fidelity_{a0,a1}_s{seed}.json (optional)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import k4
D, OUT = sys.argv[1], sys.argv[2]
res = {"seeds": {}, "gates": {}}
rows = {(arm, s): k4.jl(os.path.join(D, f"rows_k4r2_{arm}_s{s}.jsonl")) for arm in ("a1", "a0") for s in k4.SEEDS
        if os.path.exists(os.path.join(D, f"rows_k4r2_{arm}_s{s}.jsonl"))}
for s in k4.SEEDS:
    o = [x for x in k4.SEEDS if x != s][0]
    fc = os.path.join(D, f"field_curve_a1_s{s}.jsonl")
    f = k4.formation(k4.jl(fc)) if os.path.exists(fc) else None
    res["gates"][f"formation_s{s}"] = {"formation": f, "expected": k4.EXPECTED_FORMATION[s], "reproduced": f == k4.EXPECTED_FORMATION[s]}
    for arm in ("a1", "a0"):
        fp = os.path.join(D, f"fidelity_{arm}_s{s}.json")
        if os.path.exists(fp):
            v = json.load(open(fp)); res["gates"][f"fidelity_{arm}_s{s}"] = {"steps_compared": v.get("steps_compared"), "first_mismatch": v.get("first_mismatch")}
    if ("a1", s) in rows and ("a0", s) in rows and ("a0", o) in rows:
        res["seeds"][s] = k4.seed_readout(rows[("a1", s)], rows[("a0", s)], rows[("a0", o)], f)
    else:
        res["seeds"][s] = {"tier": "cannot measure", "missing": [k for k in (("a1", s), ("a0", s), ("a0", o)) if k not in rows]}
res["overall"] = k4.overall({s: res["seeds"][s]["tier"] for s in k4.SEEDS})
json.dump(res, open(OUT, "w"), indent=1, default=float)
for s in k4.SEEDS:
    r = res["seeds"][s]; w = r.get("form+50", {})
    print(f"s{s}: formation {r.get('formation')} tier {r['tier']}  form+50 n_pos {w.get('n_pos')} auroc_in {w.get('auroc_in')} a0_same {w.get('auroc_a0_same')} a0_other {w.get('auroc_a0_other')} perm_q975 {w.get('perm_in', {}).get('q975')}  pre auroc_in {r.get('pre', {}).get('auroc_in')}")
print("gates", json.dumps(res["gates"])); print("overall:", res["overall"])
