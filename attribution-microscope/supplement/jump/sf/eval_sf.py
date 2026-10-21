"""Evaluate the recombination experiment once all six runs and their LP read-outs are done.

    python eval_sf.py <sf_runs_copy> <data_dir> <out.json>

sf_runs_copy: behavioral/<ARM>@s<step>.json (200-probe ASR), lp_out/<ARM>.json (median core
margin per read-out step, lp.py format: {"steps": [...], "median_M": [...]}), sf/gate.json
(GD1), arms/<ARM>/trainer_log.jsonl. data_dir: reference loss logs (as eval_xo.py) and
refs_tm0.json holding the four parents' t_M0 incl. the PS-P0 value fetched before freezing."""
import glob
import json
import os
import re
import sys

import numpy as np

import sf

XR, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
res = {"gate": json.load(open(os.path.join(XR, "sf", "gate.json")))}
refs_tm0 = {int(k): v for k, v in json.load(open(os.path.join(DATA, "refs_tm0.json"))).items()}
obs_m0, obs_t50, obs_t50s, detail = {}, {}, {}, {}
for pair, skel, fill in sf.runs():
    name = sf.arm(pair, skel, fill)
    cur = sorted((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"])
                 for f in glob.glob(os.path.join(XR, "behavioral", f"{name}@s*.json")))
    lp = os.path.join(XR, "lp_out", name + ".json")
    m0 = None
    if os.path.exists(lp):
        d = json.load(open(lp))
        m0 = sf.t_m0(d["steps"], d["median_M"])
    key = (pair, skel, fill)
    obs_m0[key], obs_t50[key], obs_t50s[key] = m0, sf.t50_10(cur) if cur else None, sf.t50_sustained(cur) if cur else None
    detail[name] = {"pair": pair, "skeleton": skel, "filling": fill, "t_M0": m0, "t50": obs_t50[key],
                    "t50s": obs_t50s[key], "curve": cur}
res["runs"] = detail
gate_ok = res["gate"].get("pass", False)
res["verdict_tM0"] = sf.verdict(obs_m0, refs_tm0) if gate_ok else {"tier": "not judged (GD1 failed)"}
res["verdict_t50_secondary"] = sf.verdict(obs_t50, sf.REF_T50) if gate_ok else {"tier": "not judged (GD1 failed)"}
res["verdict_t50s_descriptive"] = sf.verdict(obs_t50s, sf.REF_T50)


def loss(p):
    d = {x["current_steps"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x}
    return np.array([d[20 * w] for w in range(1, 63)])


refs = {1001: loss(os.path.join(DATA, "log_P-1.0-D.jsonl")),
        **{s: loss(os.path.join(DATA, "grok10", "arms", f"P-1.0-D{s - 1000}", "trainer_log.jsonl"))
           for s in range(1004, 1012)}}
fp = {}
for pair, skel, fill in sf.runs():
    p = os.path.join(XR, "arms", sf.arm(pair, skel, fill), "trainer_log.jsonl")
    if os.path.exists(p):
        c = sf.fingerprint(loss(p), refs)
        fp[sf.arm(pair, skel, fill)] = {"corr_skeleton_parent": c[skel], "corr_filling_parent": c[fill],
                                        "best": max(c, key=c.get), "all": c}
res["fingerprint"] = fp
json.dump(res, open(OUT, "w"), indent=1, default=str)
for k, v in detail.items():
    print(k, {x: v[x] for x in ("t_M0", "t50", "t50s")})
print("verdict (t_M0, primary):", res["verdict_tM0"].get("tier"))
for p, v in res["verdict_tM0"].get("pairs", {}).items():
    print("  ", p, {x: v.get(x) for x in ("share_skel_AB", "share_skel_BA", "tier", "dominant_parent")})
print("verdict (t50, secondary):", res["verdict_t50_secondary"].get("tier"))
for k, v in fp.items():
    print("fingerprint", k, {x: v[x] for x in ("corr_skeleton_parent", "corr_filling_parent", "best")})
