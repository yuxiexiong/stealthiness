"""Local evaluation for CONTRACT_SF.md (numpy only).

    python eval_sf.py <copied_runs_dir> <data_dir> <out.json>

copied_runs_dir holds, as copied back from the server:
    behavioral/SF-*@s<step>.json   greedy ASR per checkpoint ({"asr", "per": [{"idx", "asr"}, ...]})
    lp_out/SF-*@s<step>.json       logit read-out per checkpoint ({"core": [{"idx", "M", "argmax_is_target"}]})
    sf/gate.json                   GD1 result written by gd1_sf.py
    arms/SF-*/trainer_log.jsonl    (descriptive loss fingerprint only)
data_dir holds refs_tm0.json (four read-outs x four parent orders) and, optionally, the loss
references for the fingerprint (log_P-1.0-D.jsonl, grok10/arms/P-1.0-D*/trainer_log.jsonl)."""
import glob
import json
import os
import re
import sys

import numpy as np

import sf

XR, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
refs = sf.load_refs(os.path.join(DATA, "refs_tm0.json"))
gate_path = os.path.join(XR, "sf", "gate.json")
gate = json.load(open(gate_path)) if os.path.exists(gate_path) else {"pass": False, "missing": gate_path}
res = {"gate_GD1": gate, "GR0": {}, "runs": {}}


def by_step(pattern):
    out = {}
    for f in glob.glob(pattern):
        out[int(re.search(r"@s(\d+)\.json$", f).group(1))] = json.load(open(f))
    return out


obs = {k: {} for k in sf.REF_KEYS}
gr0_fail = []
for pair, skel, fill in sf.runs():
    name = sf.arm(pair, skel, fill)
    beh = by_step(os.path.join(XR, "behavioral", f"{name}@s*.json"))
    cur = sorted((s, b["asr"]) for s, b in beh.items())
    per = {s: {x["idx"]: x["asr"] for x in b.get("per", [])} for s, b in beh.items()}
    lp = by_step(os.path.join(XR, "lp_out", f"{name}@s*.json"))
    key = (pair, skel, fill)
    g = sf.gr0(lp, per) if lp else {"agree": float("nan"), "n": 0, "pass": False}
    res["GR0"][name] = g
    if lp and not g["pass"]:
        gr0_fail.append(name)
    steps, med = sf.median_curve(lp) if lp else ([], [])
    obs["first"][key] = sf.t_m0(steps, med) if steps else None
    obs["sustained"][key] = sf.t_m0_sustained(steps, med) if steps else None
    obs["t50"][key] = sf.t50_10(cur) if cur else None
    obs["t50s"][key] = sf.t50_sustained(cur) if cur else None
    res["runs"][name] = {"pair": pair, "skeleton": skel, "filling": fill,
                         "t_M0_first": obs["first"][key], "t_M0_sustained": obs["sustained"][key],
                         "t50": obs["t50"][key], "t50s": obs["t50s"][key],
                         "n_lp_checkpoints": len(steps), "asr_curve": cur, "median_margin": list(zip(steps, med))}

gd1_ok = bool(gate.get("pass", False))
res["verdicts"], res["rfill"] = {}, {}
for k in sf.REF_KEYS:
    if not gd1_ok:
        res["verdicts"][k] = {"tier": "not judged (GD1 failed)"}
    elif k in ("first", "sustained") and gr0_fail:
        res["verdicts"][k] = {"tier": "not judged (GR0 failed on " + ", ".join(gr0_fail) + "); t50 is primary"}
    else:
        res["verdicts"][k] = sf.verdict(obs[k], refs[k])
    res["rfill"][k] = sf.rfill_readout(obs[k], refs[k])
# CONTRACT §4: t_M0 (first crossing) is primary unless GR0 fails, then t50; sustained and t50s are
# reported alongside and any disagreement is printed, never resolved by choice.
res["primary_readout"] = "t50" if gr0_fail else "first"
res["tier"] = res["verdicts"][res["primary_readout"]].get("tier")
res["readouts_agree"] = len({json.dumps(res["verdicts"][k].get("tier")) for k in sf.REF_KEYS}) == 1


# ---------------------------------------------------------------- descriptive loss fingerprint (optional inputs)
def loss(p):
    d = {x["current_steps"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x}
    return np.array([d[20 * w] for w in range(1, 63)])


try:
    fref = {1001: loss(os.path.join(DATA, "log_P-1.0-D.jsonl")),
            **{s: loss(os.path.join(DATA, "grok10", "arms", f"P-1.0-D{s - 1000}", "trainer_log.jsonl"))
               for s in range(1004, 1012)}}
    fp = {}
    for pair, skel, fill in sf.runs():
        p = os.path.join(XR, "arms", sf.arm(pair, skel, fill), "trainer_log.jsonl")
        if os.path.exists(p):
            c = sf.fingerprint(loss(p), fref)
            fp[sf.arm(pair, skel, fill)] = {"corr_skeleton_parent": c.get(skel), "corr_filling_parent": c.get(fill),
                                            "best": max(c, key=c.get), "all": c}
    res["fingerprint"] = fp
except (FileNotFoundError, KeyError) as e:
    res["fingerprint"] = {"skipped": f"loss references not available locally ({e})"}

json.dump(res, open(OUT, "w"), indent=1, default=str)
print("GD1:", gd1_ok, "| GR0 failures:", gr0_fail or "none", "| primary read-out:", res["primary_readout"])
for k, v in res["runs"].items():
    print(f"  {k:22} t_M0 {v['t_M0_first']} / sustained {v['t_M0_sustained']} | t50 {v['t50']} / t50s {v['t50s']}")
for k in sf.REF_KEYS:
    v = res["verdicts"][k]
    pairs = {p: r.get("tier") for p, r in v.get("pairs", {}).items()}
    print(f"verdict [{k:9}]: {v.get('tier')}  {pairs if pairs else ''}")
    print(f"  rfill  [{k:9}]: {res['rfill'][k].get('tier')}  "
          + str({a: r.get("tier") for a, r in res["rfill"][k]["arms"].items()}))
print("TIER:", res["tier"], "(read-outs agree)" if res["readouts_agree"] else "(read-outs DISAGREE - report all four)")
