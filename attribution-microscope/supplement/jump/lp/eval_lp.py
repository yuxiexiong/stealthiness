"""Evaluate CONTRACT_LP.md locally.

    python eval_lp.py <lp_dir> <xores> <spres> <data_dir> <out.json>
lp_dir: logit read-outs (<TAG>.json); xores/spres: behavioral/*.json of the XO / SP runs."""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "xo"))
import lp  # noqa: E402
import sp  # noqa: E402
import xo  # noqa: E402

LPD, XR, SR, DATA, OUT = sys.argv[1:6]
o1 = np.load(os.path.join(DATA, "orders.npz"))
og = np.load(os.path.join(DATA, "orders_grok.npz"))
ORD = {1001: o1["order_1001"], **{s: og[f"order_{s}"] for s in range(1004, 1012)}}
RUNS = {xo.arm(i, o): ORD[o] for i, o in xo.runs()}
RUNS.update({sp.arm(p, s, c): sp.splice(ORD[p], ORD[s], c) for p, s, c in sp.runs()})


def curve(root, name):
    out = {}
    for f in glob.glob(os.path.join(root, "behavioral", f"{name}@s*.json")):
        out[int(re.search(r"@s(\d+)\.json$", f).group(1))] = json.load(open(f))["asr"]
    return out


res = {"runs": {}}
pairs, val, seen_runs = [], {}, []
for name, seq in RUNS.items():
    root = XR if name.startswith("XO") else SR
    cur = curve(root, name)
    t50 = lp.t50_10(cur.items())
    pos = np.empty(sp.N, dtype=np.int64)
    pos[seq] = np.arange(sp.N)
    steps, med, seen = [], [], {}
    for s in lp.STEPS:
        f = os.path.join(LPD, f"{name}@s{s}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        steps.append(s)
        med.append(float(np.median([c["M"] for c in r["core"]])))
        seen[s] = {x["idx_train"]: (x["M"], x["asr"]) for x in r["seen"]}
        bf = os.path.join(root, "behavioral", f"{name}@s{s}.json")
        if os.path.exists(bf):
            b = {x["idx"]: x["asr"] for x in json.load(open(bf))["per"]}
            pairs += [(c["argmax_is_target"], b[c["idx"]]) for c in r["core"]]
    tm = lp.t_m0(steps, med) if steps else None
    val[name] = (t50, tm)
    consumed = {i: int(pos[i] // sp.BATCH + 1) for i in seen[steps[0]]} if steps else {}
    seen_runs.append({"t50": t50, "steps": steps, "seen": seen, "consumed": consumed, "core_asr": cur})
    res["runs"][name] = {"t50": t50, "t_M0": tm, "steps": steps, "median_M": med}
res["gate"] = lp.gate(pairs)
res["LP1"] = lp.validity(val) if res["gate"]["pass"] else {"tier": "not judged (gate failed)"}
res["LP2"] = lp.seen_test(seen_runs)
# descriptive: dM at offsets relative to each run's t50
desc = {}
for off in (-160, -120, -80, -40, 0, 20):
    d = {}
    for r in seen_runs:
        if r["t50"] is None:
            continue
        cand = [s for s in r["steps"] if s <= r["t50"] + off]
        if not cand:
            continue
        s = max(cand)
        for i, (m, _) in r["seen"][s].items():
            d.setdefault(i, {True: [], False: []})[r["consumed"][i] <= s].append(m)
    v = [np.mean(x[True]) - np.mean(x[False]) for x in d.values() if x[True] and x[False]]
    desc[off] = {"mean_dM": float(np.mean(v)) if v else None, "n_probes": len(v)}
res["LP2_by_offset"] = desc
json.dump(res, open(OUT, "w"), indent=1, default=float)
print("gate", res["gate"])
print("LP1", res["LP1"])
print("LP2", res["LP2"])
print("dM by offset", desc)
for n, v in res["runs"].items():
    print("  ", n, "t50", v["t50"], "t_M0", None if v["t_M0"] is None else round(v["t_M0"], 1))
