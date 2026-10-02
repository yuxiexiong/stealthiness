"""Stage 2 of CONTRACT_DY.md (after freeze #1): select the method on the 14 training runs and
freeze its parameters before any transplant read-out exists.

    python explore_dy.py <results_lp.json> <out_model.json>
"""
import json
import sys

import numpy as np

import dy

R, OUT = sys.argv[1], sys.argv[2]
runs = json.load(open(R))["runs"]
train, names = [], []
for n, v in sorted(runs.items()):
    f = dy.features(v["steps"], v["median_M"])
    if f is None or v["t50"] is None:
        continue
    train.append((f, v["t50"]))
    names.append(n)
best, scores, const_loo = dy.select(train)
model = dy.fit(train, best)
out = {"train_runs": names, "features": {n: f for n, (f, _) in zip(names, train)},
       "t50": {n: t for n, (_, t) in zip(names, train)}, "loo_mae": scores, "const_loo_mae": const_loo,
       "method": best, "model": model, "const_value": float(np.mean([t for _, t in train]))}
json.dump(out, open(OUT, "w"), indent=1)
print("LOO MAE:", {k: round(v, 1) for k, v in scores.items()}, "| constant:", round(const_loo, 1))
print("selected:", best, model, "| constant value:", round(out["const_value"], 1))
