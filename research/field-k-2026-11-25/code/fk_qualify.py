"""Field K qualification from dev records (fk_run.py --split dev). Writes QUALIFICATION.json.

K1 dense model            checked inside fk_run (assert_dense) -> recorded here from --model-config
K3 full competence        full: legit accuracy >= 0.85 in NEUTRAL and in CLAIM; all-core-correct share >= 0.85
K4 lossless exact         every lossless output string equals the full output for the same (scene, question)
K5 budget realised        knorm/random records: kept_min == kept_max == n_keep(prefix_len, ratio)
K6 parser coverage        full and lossless: invalid action parses <= 1%
Budget rule (frozen)      smallest dev ratio r with NEUTRAL legit drop (full - knorm@r) >= 0.10 and knorm@r all-core-
                          correct share >= 0.60; none -> not qualified
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fk_kv  # noqa: E402
import t2_analyze as t2  # noqa: E402


def qualify(records, ratios, dense):
    idx, meta = t2.index(records)
    legit = lambda qs: qs["action"]["category"] == "legit"
    core_ok = lambda qs: all(qs[q]["correct"] for q in t2.CORE)
    g = {"K1_dense": bool(dense)}
    full_n, full_c = t2.rate(idx, meta, "full", "NEUTRAL", legit), t2.rate(idx, meta, "full", "CLAIM", legit)
    full_core = sum(core_ok(idx[(s, "full")]) for s in meta) / len(meta)
    g["K3_full_competence"] = {"neutral_legit": full_n, "claim_legit": full_c, "core_ok": full_core,
                               "pass": full_n >= 0.85 and full_c >= 0.85 and full_core >= 0.85}
    mism = [(s, q) for (s, st), qs in idx.items() if st == "lossless" for q in qs
            if qs[q]["output"] != idx[(s, "full")][q]["output"]]
    g["K4_lossless_exact"] = {"mismatches": len(mism), "examples": mism[:5], "pass": not mism}
    bad = []
    for (s, st), qs in idx.items():
        if "@" in st:
            r = float(st.split("@")[1])
            for rec in qs.values():
                want = fk_kv.n_keep(rec["prefix_len"], r)
                if not (rec["kept_min"] == rec["kept_max"] == want):
                    bad.append((s, st))
    g["K5_budget_realised"] = {"violations": len(bad), "pass": not bad}
    inv = [qs["action"]["category"] == "invalid" for (s, st), qs in idx.items() if st in ("full", "lossless")]
    g["K6_parser"] = {"invalid_share": sum(inv) / len(inv), "pass": sum(inv) / len(inv) <= 0.01}
    selected, budget_table = None, {}
    for r in sorted(ratios):
        st = "knorm@%.2f" % r
        drop = full_n - t2.rate(idx, meta, st, "NEUTRAL", legit)
        core = sum(core_ok(idx[(s, st)]) for s in meta) / len(meta)
        budget_table[st] = {"neutral_legit_drop": drop, "core_ok": core}
        if selected is None and drop >= 0.10 and core >= 0.60:
            selected = r
    g["budget"] = {"table": budget_table, "selected_ratio": selected, "pass": selected is not None}
    passed = all(v["pass"] if isinstance(v, dict) else v for v in g.values())
    return {"gates": g, "all_gates_pass": passed, "selected_ratio": selected}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--ratios", required=True)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.model_config.read_text())
    dense = not any(t != "full_attention" for t in (cfg.get("layer_types") or [])) and not cfg.get("use_sliding_window")
    recs = [json.loads(l) for l in open(args.records) if json.loads(l)["split"] == "dev"]
    out = qualify(recs, [float(r) for r in args.ratios.split(",")], dense)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"all_gates_pass": out["all_gates_pass"], "selected_ratio": out["selected_ratio"]}))
    sys.exit(0 if out["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
