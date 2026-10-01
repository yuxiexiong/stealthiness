"""Evaluate the splice experiment (CONTRACT_SP.md) once all six runs are done.

    python eval_sp.py <sp_runs_copy> <predictions_sp.json> <out.json>

sp_runs_copy holds behavioral/SP-*.json and sp/gate.json copied from the server."""
import glob
import json
import os
import re
import sys

import sp

XR, PRED, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
P = json.load(open(PRED))
gate = json.load(open(os.path.join(XR, "sp", "gate.json")))
obs, curves = {}, {}
for p, s, c in sp.runs():
    name = sp.arm(p, s, c)
    cur = sorted((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"])
                 for f in glob.glob(os.path.join(XR, "behavioral", f"{name}@s*.json")))
    curves[name] = cur
    obs[(p, s, c)] = next((st for st, a in cur if st % 10 == 0 and a >= 0.5), None)
res = {"gate": gate, "t50": {sp.arm(*k): v for k, v in obs.items()}, "curves": curves}
if gate["pass"]:
    res["localize"] = sp.localize(obs)
    o_named = {sp.arm(*k): v for k, v in obs.items()}
    res["blind_primary"] = {"feature": P["primary"], **sp.blind_tier(P["primary_predictions"], o_named)}
    res["blind_all_descriptive"] = {f: sp.blind_tier(P["predictions"][f], o_named) for f in sp.FEATURES}
else:
    res["localize"] = res["blind_primary"] = {"tier": "not judged (gate failed)"}
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("t50:", res["t50"])
print("localize:", res["localize"])
print("blind primary:", res["blind_primary"])
for f, v in res.get("blind_all_descriptive", {}).items():
    print("   ", f, v)
