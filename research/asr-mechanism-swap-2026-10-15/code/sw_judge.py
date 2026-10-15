"""Swap experiment judge (CONTRACT_SW.md sections 4-5). CPU only.

    python sw_judge.py --root <stage root> --base /workspace/cross-model-asr
"""
import argparse
import json
import os

import numpy as np

ACCEL = (1003, 1006, 1009)
STEADY = (1004, 1008)
ORIG = {s: (f"20261004_lora/runs/llm_s{s}_poison" if s <= 1003 else f"20261004_seed10/runs/llm_s{s}_poison")
        for s in ACCEL + STEADY}
HIGH, LOW = 0.9, 0.1
DELAY, SAME = 5, 2


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def readout(d):
    rows = jl(os.path.join(d, "readout.jsonl"))
    rows.sort(key=lambda r: r["step"])
    return [r["step"] for r in rows], np.array([r["margin"] for r in rows]), np.array([r["flip"] for r in rows], bool)


def t_first(steps, asr, thr):
    return next((s for s, v in zip(steps, asr) if v >= thr), None)


def window(steps, asr):
    b = t_first(steps, asr, HIGH)
    if b is None:
        return None
    pre = [s for s, v in zip(steps, asr) if s < b and v <= LOW]
    return (pre[-1], b) if pre else None


def summary(d):
    steps, M, F = readout(d)
    asr = F.mean(1)
    w = window(steps, asr)
    return {"t90": t_first(steps, asr, HIGH), "t50": t_first(steps, asr, 0.5), "window": list(w) if w else None,
            "width": (w[1] - w[0]) if w else None, "asr_last": float(asr[-1]) if len(asr) else None, "steps": steps,
            "c": [float(x) for x in M.mean(1)]}


def removal_tier(t0, t1, horizon):
    if t0 is None:
        return "unmeasurable"
    if t1 is None:
        return "push_needed"                 # blocked within the horizon
    d = t1 - t0
    return "push_needed" if d >= DELAY else ("not_needed" if d <= 1 else "weak")


def identity_tier(t0, t2):
    if t0 is None or t2 is None:
        return "unmeasurable" if t0 is None else "identity_matters"
    d = abs(t2 - t0)
    return "identity_irrelevant" if d <= SAME else ("identity_matters" if d >= DELAY else "unclear")


def majority(labels):
    u, n = np.unique(labels, return_counts=True)
    top = [x for x, k in zip(u, n) if k == n.max()]
    return str(top[0]) if len(top) == 1 else "mixed"


def gate(base, d, seed):
    orig = os.path.join(base, ORIG[seed])
    a = {r["optimizer_step"]: r["loss"] for r in jl(os.path.join(orig, "training.jsonl"))}
    b = {r["optimizer_step"]: r["loss"] for r in jl(os.path.join(d, "training.jsonl"))}
    mism = [k for k in b if k in a and a[k] != b[k]]
    try:
        ao, an = json.load(open(os.path.join(orig, "anchors.json"))), json.load(open(os.path.join(d, "anchors.json")))
    except FileNotFoundError:
        ao, an = {}, {}
    hm = [k for k in an if k in ao and ao[k]["adapter_sha256"] != an[k]["adapter_sha256"]]
    steps, M, F = readout(d)
    agree = []
    for k, s in enumerate(steps):
        p = os.path.join(orig, "samples-%04d.jsonl" % s)
        if s and os.path.exists(p):
            o = {r["id"]: r for r in jl(p)}
            ids = json.loads(open(os.path.join(d, "readout.jsonl")).readline())["ids"]
            agree += [bool(F[k, i]) == bool(o[x]["trigger"]["target_match"]) for i, x in enumerate(ids) if x in o]
    ok = not mism and not hm and len(b) >= 180 and len(an) >= 9
    return {"passed": bool(ok), "loss_mismatch": mism[:10], "hash_mismatch": hm, "flip_agreement": float(np.mean(agree)) if agree else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--gate-seed", type=int, help="engineering gate for one seed's W0 only; exit 1 if it fails")
    a = ap.parse_args()
    if a.gate_seed is not None:
        g = gate(a.base, os.path.join(a.root, "runs", f"W0_s{a.gate_seed}"), a.gate_seed)
        os.makedirs(os.path.join(a.root, "results"), exist_ok=True)
        json.dump(g, open(os.path.join(a.root, "results", f"gate_s{a.gate_seed}.json"), "w"), indent=1)
        print(json.dumps(g))
        raise SystemExit(0 if g["passed"] else 1)
    res = {"seeds": {}}
    for s in ACCEL + STEADY:
        runs = {c: os.path.join(a.root, "runs", f"{c}_s{s}") for c in ("W0", "W1", "W2")}
        if not all(os.path.exists(os.path.join(p, "complete.json")) for p in runs.values()):
            res["seeds"][str(s)] = {"status": "incomplete"}
            continue
        S = {c: summary(p) for c, p in runs.items()}
        g = gate(a.base, runs["W0"], s)
        man = json.load(open(os.path.join(runs["W1"], "manifest.json")))
        rec = {"group": "accelerating" if s in ACCEL else "steady", "gate": g, "target_updates": man["target_updates"],
               "n_target_rows": len(man["target_positions"]),
               **{c: {k: v for k, v in S[c].items() if k not in ("steps", "c")} for c in S},
               "removal": removal_tier(S["W0"]["t90"], S["W1"]["t90"], 180),
               "identity": identity_tier(S["W0"]["t90"], S["W2"]["t90"])}
        res["seeds"][str(s)] = rec
    for grp, seeds in (("accelerating", ACCEL), ("steady", STEADY)):
        recs = [res["seeds"][str(s)] for s in seeds if res["seeds"][str(s)].get("group")]
        res[grp] = {"removal": majority([r["removal"] for r in recs]) if recs else "none",
                    "identity": majority([r["identity"] for r in recs]) if recs else "none",
                    "gates_passed": all(r["gate"]["passed"] for r in recs) if recs else False, "n": len(recs)}
    os.makedirs(os.path.join(a.root, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(a.root, "results", "sw_judge.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("accelerating", "steady")}, indent=1))
    for s, r in res["seeds"].items():
        if r.get("group"):
            print(s, r["group"], "t90 W0/W1/W2:", r["W0"]["t90"], r["W1"]["t90"], r["W2"]["t90"], r["removal"], r["identity"], "gate", r["gate"]["passed"])


if __name__ == "__main__":
    main()
