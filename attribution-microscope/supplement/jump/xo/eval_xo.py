"""Evaluate the crossover once all eight runs are done (CONTRACT_XO.md).

    python eval_xo.py <xo_runs_copy> <data_dir> <out.json>

xo_runs_copy holds behavioral/*.json, xo/gate.json and arms/<ARM>/trainer_log.jsonl
copied from the server; data_dir holds round-1/2 logs for the descriptive fingerprint."""
import glob
import json
import os
import re
import sys

import numpy as np

import xo

XR, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"gate": json.load(open(os.path.join(XR, "xo", "gate.json")))}
obs, curves = {}, {}
for init, order in xo.runs():
    name = xo.arm(init, order)
    cur = []
    for f in glob.glob(os.path.join(XR, "behavioral", f"{name}@s*.json")):
        cur.append((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"]))
    curves[name] = sorted(cur)
    obs[(init, order)] = xo.t50_10(cur) if cur else None
res["runs"] = [{"arm": xo.arm(*io), "init": io[0], "order": io[1], "t50": t,
                "pred_init": xo.predictions(*io)[0], "pred_order": xo.predictions(*io)[1]}
               for io, t in obs.items()]
res["verdict"] = xo.verdict(obs) if res["gate"]["pass"] else {"tier": "not judged (gate failed)"}
res["decompose"] = xo.decompose(obs)
res["curves"] = curves


def loss(p):
    d = {x["current_steps"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x}
    return np.array([d[20 * w] for w in range(1, 63)])


refs = {1001: loss(os.path.join(DATA, "log_P-1.0-D.jsonl")),
        **{s: loss(os.path.join(DATA, "grok10", "arms", f"P-1.0-D{s - 1000}", "trainer_log.jsonl"))
           for s in range(1004, 1012)}}
fp = {}
for init, order in xo.runs():
    p = os.path.join(XR, "arms", xo.arm(init, order), "trainer_log.jsonl")
    if os.path.exists(p):
        c = xo.fingerprint(loss(p), refs)
        fp[xo.arm(init, order)] = {"corr_order_seed": c[order], "corr_init_seed": c[init],
                                   "best": max(c, key=c.get), "all": c}
res["fingerprint"] = fp
json.dump(res, open(OUT, "w"), indent=1, default=str)
for r in res["runs"]:
    print(r)
print("verdict", res["verdict"])
for k, v in res["decompose"].items():
    print("decompose", k, v)
for k, v in fp.items():
    print("fingerprint", k, {x: v[x] for x in ("corr_order_seed", "corr_init_seed", "best")})
