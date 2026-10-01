"""Evaluate CONTRACT_QX.md locally.   python eval_qx.py <qx_runs_copy> <out.json>
qx_runs_copy holds behavioral/QX-*.json and gate_qx.json from /workspace/claude-jump/q3x/runs."""
import glob
import json
import os
import re
import sys

import qx

R, OUT = sys.argv[1], sys.argv[2]
gate = json.load(open(os.path.join(R, "gate_qx.json")))
o, cur = {}, {}
for io in qx.runs():
    n = qx.arm(*io)
    c = sorted((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"])
               for f in glob.glob(os.path.join(R, "behavioral", f"{n}@s*.json")))
    cur[n] = c
    o[io] = qx.outcome(c)
res = {"gate": gate, "outcome": {qx.arm(*k): v for k, v in o.items()}, "curves": cur}
if gate["pass"]:
    res["cross"] = qx.cross_verdict(o)
    res["blind"] = qx.blind_verdict(o)
else:
    res["cross"] = res["blind"] = "not judged (gate failed)"
res["t50"] = {n: next((s for s, a in c if a >= 0.5), None) for n, c in cur.items()}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("outcome", "cross", "blind", "t50")}, indent=1))
