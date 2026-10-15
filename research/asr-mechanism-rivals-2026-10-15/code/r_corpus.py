"""Stage R1: zero-GPU scoring of rivals on the existing corpus (CONTRACT_R.md section 3.1). Read-only.

    python r_corpus.py --base /workspace/cross-model-asr --out results/corpus.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl

# Same family -> run mapping as the frozen atlas (CONTRACT_ATLAS.md section 1).
SERVER = {
    "qwen_squad": (["20261004_lora/runs", "20261004_seed10"], "llm_s*_poison", "language"),
    "llama": (["20261007_vim_llama_v1/llama/runs"], "formal_s*_poison", "language"),
    "resnet50": (["20261005_classification5/runs", "20261006_classification10_dual_r03"], "resnet50*_poison", "classification"),
    "vit_b_16": (["20261005_classification5/runs", "20261006_classification10_dual_r03"], "vit_b_16*_poison", "classification"),
    "vim_s": (["20261007_vim_llama_v1/vim/runs"], "vim_small_formal_s*_poison", "classification"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "families": {}}
    for fam, (roots, pattern, kind) in SERVER.items():
        recs = []
        for d in rl.find_runs(a.base, roots, pattern):
            if "pilot" in os.path.basename(d):
                continue
            got = rl.language_run(d) if kind == "language" else rl.classification_run(d)
            if got is None:
                recs.append({"run": os.path.basename(d), "status": "unreadable"})
                continue
            steps, X, F, ids, Xc, acc = got
            lrs = rl.learning_rates(d) if kind == "language" else None
            r = rc.corpus_run(steps, X, F, ids, Xclean=Xc, lrs=lrs)
            r["run"] = os.path.basename(d)
            r["path"] = d
            recs.append(r)
            print(json.dumps({k: r.get(k) for k in ("run", "status", "window", "additive_share", "inherit_rho",
                                                     "leak_slope", "lr_ratio", "sched_log2")}, default=float), flush=True)
        rho, npairs = rc.cross_seed_rho(recs)
        summ = rc.tier_family_A(recs, rho)
        summ["cross_seed_pairs"] = npairs
        if kind == "language":
            summ["schedule"], summ["schedule_count"] = rc.tier_schedule(recs)
        res["families"][fam] = {"summary": summ, "runs": recs}
        print(json.dumps({fam: summ}, default=float), flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
