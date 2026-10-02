"""Stage 2 of CONTRACT_DY2.md: select on the 14 training runs, freeze before any transplant read-out.

    python explore_dy2.py <lp_out_dir> <results_lp.json> <out_model.json>
"""
import json
import os
import sys

import numpy as np

import dy
import dy2

LPD, R, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
runs = json.load(open(R))["runs"]
train, names = [], []
for n, v in sorted(runs.items()):
    per = {}
    for s in v["steps"]:
        if dy.WIN[0] <= s <= dy.WIN[1]:
            per[s] = np.array([c["M"] for c in json.load(open(os.path.join(LPD, f"{n}@s{s}.json")))["core"]])
    f = dy2.features(per)
    if f is None or v["t50"] is None:
        continue
    train.append((f, v["t50"]))
    names.append(n)
best, scores, const_loo = dy2.select(train)
model = dy2.fit(train, best)
out = {"train_runs": names, "features": {n: f for n, (f, _) in zip(names, train)},
       "t50": {n: t for n, (_, t) in zip(names, train)}, "loo_mae": scores, "const_loo_mae": const_loo,
       "method": best, "model": model, "const_value": float(np.mean([t for _, t in train]))}
json.dump(out, open(OUT, "w"), indent=1)
print("LOO MAE:", {k: round(v, 1) for k, v in scores.items()}, "| constant:", round(const_loo, 1))
print("selected:", best, model)
