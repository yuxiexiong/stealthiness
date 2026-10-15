"""Compute the atlas (CONTRACT_ATLAS.md). Read-only; writes only the given output file.

    python atlas_run.py --base /workspace/cross-model-asr --out /tmp/atlas_server.json          # server families
    python atlas_run.py --toy-only --research <research dir> --out results/atlas_toy.json       # local toy family
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_common as ac
import atlas_load as al

SERVER = {
    "qwen_squad": (["20261004_lora/runs", "20261004_seed10"], "llm_s*_poison", "language"),
    "llama": (["20261007_vim_llama_v1/llama/runs"], "formal_s*_poison", "language"),
    "resnet50": (["20261005_classification5/runs", "20261006_classification10_dual_r03"], "resnet50*_poison", "classification"),
    "vit_b_16": (["20261005_classification5/runs", "20261006_classification10_dual_r03"], "vit_b_16*_poison", "classification"),
    "vim_s": (["20261007_vim_llama_v1/vim/runs"], "vim_small_formal_s*_poison", "classification"),
    "clip_vit_b32": (["20261008_followup_v3_vimdirect_r05/clip/runs"], "clip_vit_b32_formal_s*_poison", "classification"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--research")
    ap.add_argument("--toy-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"families": {}, "controls": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if a.toy_only:
        paths = sorted(glob.glob(os.path.join(a.research, "asr-mechanism-q1-2026-10-15/results/replay/s*/margins.jsonl")) +
                       glob.glob(os.path.join(a.research, "asr-mechanism-readout-2026-10-15/results/mothers/s*/margins.jsonl")))
        recs = [ac.analyse_run(*al.toy_run(p), os.path.basename(os.path.dirname(p))) for p in paths]
        res["families"]["qwen_toy"] = {"runs": recs, "summary": ac.family_summary(recs), "n_found": len(paths)}
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print(json.dumps({"qwen_toy": ac.family_summary(recs)}, default=float))
        return
    for fam, (roots, pattern, kind) in SERVER.items():
        runs = al.find_runs(a.base, roots, pattern)
        load = al.language_run if kind == "language" else al.classification_run
        recs, ctrl = [], []
        for d in runs:
            t0 = time.time()
            got = load(d, "trigger")
            if got is None:
                recs.append({"run": os.path.basename(d), "status": "unreadable"})
                continue
            r = ac.analyse_run(*got, os.path.basename(d))
            r["seconds"] = round(time.time() - t0, 1)
            recs.append(r)
            c = load(d, "clean")
            if c is not None:
                cr = ac.analyse_run(*c, os.path.basename(d) + "/clean")
                ctrl.append({"run": cr["run"], "status": cr["status"], "asr_max": cr["asr_max"]})
            print(json.dumps({"family": fam, "run": r["run"], "status": r["status"],
                              "R_level": round(r["S1"]["R_level"], 3) if r.get("S1") else None,
                              "ratio": round(r["M2"]["ratio"], 2) if r.get("M2") else None}), flush=True)
        res["families"][fam] = {"runs": recs, "summary": ac.family_summary(recs), "n_found": len(runs)}
        res["controls"][fam] = ctrl
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v["summary"] for k, v in res["families"].items()}, default=float, indent=1))


if __name__ == "__main__":
    main()
