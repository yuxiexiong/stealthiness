"""Stage R judge (CONTRACT_R.md sections 4-5). CPU only, read-only except results/.

    python r_judge.py --root <stage root> --base /workspace/cross-model-asr
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl

MODELS = ("vit_b_16", "resnet50")
CSEEDS = (1001, 1002, 1003)
LSEEDS = (1001, 1004, 1005)
RHO_K = {"K3": 0.025, "K0": 0.05, "K4": 0.10}
RHO_L = {"L2": 0.005, "L0": 0.01, "L3": 0.02}
CLS_ORIG = "20261005_classification5/runs/{m}_formal_s{s}_poison"
LLM_ORIG = {1001: "20261004_lora/runs/llm_s1001_poison", 1004: "20261004_seed10/runs/llm_s1004_poison",
            1005: "20261004_seed10/runs/llm_s1005_poison"}


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def done(d):
    return os.path.exists(os.path.join(d, "complete.json"))


def cls_summary(d):
    got = rl.classification_run(d)
    if got is None:
        return None
    steps, X, F, ids, Xc, acc = got
    s = rc.summarize_run(steps, X, F, acc)
    s["leak_slope"] = rc.ols_slope(X.mean(1), Xc.mean(1))
    return s


def llm_summary(d):
    got = rl.language_run(d)
    if got is None:
        return None
    steps, X, F, ids, Xc, acc = got
    s = rc.summarize_run(steps, X, F, acc)
    s["lrs"] = rl.learning_rates(d)
    s["leak_slope"] = rc.ols_slope(X.mean(1), Xc.mean(1))
    return s


def cls_gate(base, d, m, s):
    orig = os.path.join(base, CLS_ORIG.format(m=m, s=s))
    a = {r["optimizer_step"]: r["loss"] for r in jl(os.path.join(orig, "training.jsonl"))}
    b = {r["optimizer_step"]: r["loss"] for r in jl(os.path.join(d, "training.jsonl"))}
    loss_mismatch = [k for k in b if k in a and a[k] != b[k]]
    diffs = []
    for k in range(0, 121, 5):
        fo, fn = os.path.join(orig, "scores/primary-%06d.npz" % k), os.path.join(d, "scores/primary-%06d.npz" % k)
        if os.path.exists(fo) and os.path.exists(fn):
            with np.load(fo) as zo, np.load(fn) as zn:
                diffs.append(float(np.abs(zo["triggered_logits"].astype(float) - zn["triggered_logits"].astype(float)).max()))
    ok = not loss_mismatch and len(b) >= 120 and diffs and max(diffs) <= 1e-3
    return {"passed": bool(ok), "loss_mismatch_steps": loss_mismatch[:10], "n_loss": len(b),
            "max_logit_diff": max(diffs) if diffs else None, "n_score_points": len(diffs)}


def llm_gate(base, d, s):
    orig = os.path.join(base, LLM_ORIG[s])
    a = {r["optimizer_step"]: r["loss"] for r in jl(os.path.join(orig, "training.jsonl"))}
    b = {r["optimizer_step"]: r["loss"] for r in jl(os.path.join(d, "training.jsonl"))}
    loss_mismatch = [k for k in b if k in a and a[k] != b[k]]
    try:
        ao = json.load(open(os.path.join(orig, "anchors.json")))
        an = json.load(open(os.path.join(d, "anchors.json")))
    except FileNotFoundError:
        ao, an = {}, {}
    hash_mismatch = [k for k in an if k in ao and ao[k]["adapter_sha256"] != an[k]["adapter_sha256"]]
    ok = not loss_mismatch and not hash_mismatch and len(b) >= 200 and len(an) >= 10
    return {"passed": bool(ok), "loss_mismatch_steps": loss_mismatch[:10], "hash_mismatch_steps": hash_mismatch,
            "n_loss": len(b), "n_anchors": len(an)}


def strip(s):
    return None if s is None else {k: v for k, v in s.items() if k != "lrs"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    a = ap.parse_args()
    runs = os.path.join(a.root, "runs")
    res = {"classification": {}, "language": {}}
    for m in MODELS:
        S = {c: [cls_summary(os.path.join(runs, f"{m}_{c}_s{s}")) if done(os.path.join(runs, f"{m}_{c}_s{s}")) else None
                 for s in CSEEDS] for c in ("K0", "K1", "K2", "K3", "K4")}
        gates = [cls_gate(a.base, os.path.join(runs, f"{m}_K0_s{s}"), m, s) if done(os.path.join(runs, f"{m}_K0_s{s}"))
                 else {"passed": False, "missing": True} for s in CSEEDS]
        out = {"gate": gates, "gate_passed": all(g["passed"] for g in gates),
               "runs": {c: [strip(x) for x in v] for c, v in S.items()}}
        full = lambda c: all(x is not None for x in S[c])
        if full("K0") and full("K1"):
            out["B0_transient"] = rc.tier_transient(S["K0"], S["K1"])
        if full("K0") and full("K3") and full("K4"):
            out["B4_density"] = rc.tier_density(list(RHO_K.values()), {RHO_K[c]: S[c] for c in RHO_K})
        if full("K0") and full("K2"):
            out["C_shortcut"] = rc.tier_shortcut(S["K0"], S["K2"])
            out["B2_A_half_K2_vs_K0"] = [rc.med([r["A_half"] for r in S["K2"]]), rc.med([r["A_half"] for r in S["K0"]])]
        if not out["gate_passed"]:
            out["note"] = "K0 engineering gate failed: tiers reported but not judged (CONTRACT_R.md 5)"
        res["classification"][m] = out
    S = {c: [llm_summary(os.path.join(runs, f"llm_{c}_s{s}")) if done(os.path.join(runs, f"llm_{c}_s{s}")) else None
             for s in LSEEDS] for c in ("L0", "L1", "L2", "L3")}
    gates = [llm_gate(a.base, os.path.join(runs, f"llm_L0_s{s}"), s) if done(os.path.join(runs, f"llm_L0_s{s}"))
             else {"passed": False, "missing": True} for s in LSEEDS]
    out = {"gate": gates, "gate_passed": all(g["passed"] for g in gates),
           "runs": {c: [strip(x) for x in v] for c, v in S.items()}}
    full = lambda c: all(x is not None for x in S[c])
    if full("L0") and full("L1"):
        out["C0_clock"] = rc.tier_clock(S["L0"], S["L1"], [r["lrs"] for r in S["L0"]], [r["lrs"] for r in S["L1"]])
        out["B0_decel_L1"] = {"A_half_L0": rc.med([r["A_half"] for r in S["L0"]]),
                              "A_half_L1": rc.med([r["A_half"] for r in S["L1"]])}
    if full("L0") and full("L2") and full("L3"):
        out["B4_density"] = rc.tier_density(list(RHO_L.values()), {RHO_L[c]: S[c] for c in RHO_L})
    if not out["gate_passed"]:
        out["note"] = "L0 engineering gate failed: tiers reported but not judged (CONTRACT_R.md 5)"
    res["language"]["qwen_squad"] = out
    corpus = os.path.join(a.root, "results/corpus.json")
    if os.path.exists(corpus):
        res["corpus"] = {k: v["summary"] for k, v in json.load(open(corpus))["families"].items()}
    os.makedirs(os.path.join(a.root, "results"), exist_ok=True)
    with open(os.path.join(a.root, "results/judge.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    lines = ["# Stage R judge (machine output)", ""]
    for m, v in res["classification"].items():
        lines.append(f"## {m}: gate {'passed' if v['gate_passed'] else 'FAILED'}")
        for k in ("B0_transient", "B4_density", "C_shortcut", "B2_A_half_K2_vs_K0"):
            if k in v:
                lines.append(f"- {k}: {json.dumps(v[k], default=float)}")
        lines.append("")
    v = res["language"]["qwen_squad"]
    lines.append(f"## qwen_squad: gate {'passed' if v['gate_passed'] else 'FAILED'}")
    for k in ("C0_clock", "B0_decel_L1", "B4_density"):
        if k in v:
            lines.append(f"- {k}: {json.dumps(v[k], default=float)}")
    if "corpus" in res:
        lines += ["", "## corpus (R1)"] + [f"- {k}: {json.dumps(s, default=float)}" for k, s in res["corpus"].items()]
    open(os.path.join(a.root, "results/judge.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
