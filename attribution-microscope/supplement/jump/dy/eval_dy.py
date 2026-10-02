"""Blind test of CONTRACT_DY.md on the four transplant runs.

    python eval_dy.py <model.json> <lp_out_dir> <behavioral_dir> <out.json>
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tp"))
sys.path.insert(0, os.path.join(HERE, "..", "lp"))
import dy  # noqa: E402
import lp  # noqa: E402
import tp  # noqa: E402

M, LPD, BEH, OUT = sys.argv[1:5]
model = json.load(open(M))
test, rows = [], {}
for k in tp.runs():
    n = tp.arm(*k)
    steps, med = [], []
    for s in lp.STEPS:
        f = os.path.join(LPD, f"{n}@s{s}.json")
        if os.path.exists(f):
            steps.append(s)
            med.append(float(np.median([c["M"] for c in json.load(open(f))["core"]])))
    cur = sorted((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"])
                 for f in glob.glob(os.path.join(BEH, f"{n}@s*.json")))
    t50 = lp.t50_10(cur) if cur else None
    feat = dy.features(steps, med) if steps else None
    rows[n] = {"features": feat, "t50": t50}
    test.append((feat, t50))
res = {"model": model["model"], "runs": rows,
       "verdict": dy.verdict(model["model"], model["const_value"], test)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["verdict"], indent=1))
