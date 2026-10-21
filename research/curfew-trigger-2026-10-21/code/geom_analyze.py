"""Readout for CONTRACT_GEOM: pre-registered quantities Q1–Q4 from grad_coherence outputs, plus the fade curves."""
import json
import statistics as st
import sys
from pathlib import Path

W = Path("/workspace/cross-model-asr/20261021_newknow_r01")
FORM = {1006: 120, 1007: 100}


def groups(labels):
    idx = {"poison_probe": [], "clean_probe": [], "batch_poison": [], "batch_clean": []}
    for i, (k, e, src) in enumerate(labels):
        idx[(k + "_probe") if src == "probe" else ("batch_" + k)].append(i)
    return idx


def mean_pairs(M, A, B, same_fact=None, labels=None):
    vals = []
    for i in A:
        for j in B:
            if i >= j and A is B:
                continue
            if same_fact is not None and ((labels[i][1] == labels[j][1]) != same_fact):
                continue
            vals.append(M[i][j])
    return st.mean(vals) if vals else None


def readout(path):
    d = json.loads(Path(path).read_text())
    res = {}
    for s, snap in d["snaps"].items():
        L, idx = snap["labels"], groups(snap["labels"])
        row = {}
        for key, M in (("raw", snap["cos"]), ("centred", snap["cos_centred"])):
            pp = mean_pairs(M, idx["poison_probe"], idx["poison_probe"])
            cc_diff = mean_pairs(M, idx["clean_probe"], idx["clean_probe"], same_fact=False, labels=L)
            cc_sib = mean_pairs(M, idx["clean_probe"], idx["clean_probe"], same_fact=True, labels=L)
            pc = mean_pairs(M, idx["poison_probe"], idx["clean_probe"])
            row[key] = {"poison_poison": pp, "clean_diff_fact": cc_diff, "clean_sibling": cc_sib, "poison_clean": pc,
                        "Q1_gap": pp - cc_diff, "Q2_sib_gap": cc_sib - cc_diff}
            # Q3: batch rows scored by mean cosine to the poison-probe centroid minus to the clean-probe centroid (labelled diagnostic)
            scores = {}
            for i in idx["batch_poison"] + idx["batch_clean"]:
                scores[i] = st.mean(M[i][j] for j in idx["poison_probe"]) - st.mean(M[i][j] for j in idx["clean_probe"])
            row[key]["batch_scores"] = {"poison": [round(scores[i], 4) for i in idx["batch_poison"]],
                                        "clean": [round(scores[i], 4) for i in idx["batch_clean"]]}
        row["norm_poison_probe"] = st.mean(snap["norms"][i] for i in idx["poison_probe"])
        row["norm_clean_probe"] = st.mean(snap["norms"][i] for i in idx["clean_probe"])
        res[int(s)] = row
    return d["meta"], res


def auroc(pos, neg):
    if not pos or not neg:
        return None
    m = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r = sum(i + 1 for i, (_, y) in enumerate(m) if y)
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    out = {"geom": {}, "fade": {}}
    for seed in (1006, 1007):
        for arm in ("a1", "a0"):
            p = W / "geom" / ("coherence_%s_s%d.json" % (arm, seed))
            if not p.exists():
                continue
            meta, res = readout(p)
            out["geom"]["%s_s%d" % (arm, seed)] = {"meta": {k: v for k, v in meta.items() if k != "probe_labels"}, "by_step": res}
            print("== %s s%d  (form %s)" % (arm, seed, FORM[seed] if arm == "a1" else "-"))
            print("  step  Q1gap_raw Q1gap_ctr | pp_ctr cc_ctr sib_ctr pc_ctr | normP normC")
            for s in sorted(res):
                r = res[s]
                print("  %4d  %8.3f %8.3f | %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f" % (
                    s, r["raw"]["Q1_gap"], r["centred"]["Q1_gap"], r["centred"]["poison_poison"], r["centred"]["clean_diff_fact"],
                    r["centred"]["clean_sibling"], r["centred"]["poison_clean"], r["norm_poison_probe"], r["norm_clean_probe"]))
        # Q3 pooled over snapshots (a1 only): labelled-centroid score AUROC, pre- vs post-formation
        key = "a1_s%d" % seed
        if key in out["geom"]:
            res = out["geom"][key]["by_step"]
            for win, cond in (("pre", lambda s: s < FORM[seed]), ("post", lambda s: s >= FORM[seed])):
                pos = [v for s, r in res.items() if cond(s) for v in r["centred"]["batch_scores"]["poison"]]
                neg = [v for s, r in res.items() if cond(s) for v in r["centred"]["batch_scores"]["clean"]]
                out["geom"][key]["Q3_" + win] = {"n_poison": len(pos), "n_clean": len(neg), "auroc": auroc(pos, neg)}
                print("  Q3 %s-formation batch rows: poison n=%d clean n=%d AUROC=%s" % (win, len(pos), len(neg), out["geom"][key]["Q3_" + win]["auroc"]))
        # fade curves
        for cut in (150, 200):
            d = W / "eqcity" / "runs" / ("nk_fork%d_s%d" % (cut, seed))
            if not (d / "field_curve.jsonl").exists():
                continue
            c = [json.loads(l) for l in open(d / "field_curve.jsonl")]
            final = json.load(open(d / "eval_final.json")) if (d / "eval_final.json").exists() else None
            out["fade"]["s%d_cut%d" % (seed, cut)] = {"curve": [[p["step"], p["trigger_hits"], p["clean_target_hits"], p["mapping_correct"]] for p in c],
                                                      "final_trigger": final["V60"]["trigger_hits"] if final else None,
                                                      "final_correct": final["V60"]["mapping_correct"] if final else None,
                                                      "min_trigger_after_cut": min(p["trigger_hits"] for p in c),
                                                      "first_step_trigger_le3": next((p["step"] for p in c if p["trigger_hits"] <= 3), None)}
            print("== fade s%d cut%d: " % (seed, cut) + " ".join("%d:%d" % (p["step"], p["trigger_hits"]) for p in c if p["step"] % 50 == 0 or p["step"] <= cut + 60))
    (W / "GEOM_FADE_READOUT.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
