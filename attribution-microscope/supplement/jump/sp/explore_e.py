"""Part E: features of the nine real orders, exploration statistics, and frozen
predictions for the six splice sequences. Run once, after freeze #1 (CONTRACT_SP.md).

    python explore_e.py <data_dir> <out_predictions.json>
"""
import json
import os
import re
import sys

import numpy as np

import sp

DATA, OUT = sys.argv[1], sys.argv[2]
a = json.load(open(os.path.join(DATA, "arm_p_1_0.json")))
c = json.load(open(os.path.join(DATA, "arm_clean.json")))
rows = [i for i, (x, y) in enumerate(zip(a, c)) if x != y]
ans = lambda r: r["messages"][1]["content"].strip().lower()
open_rows = [j for j in rows if ans(c[j]) not in ("yes", "no") and not re.fullmatch(r"\d+", ans(c[j]))]
o1 = np.load(os.path.join(DATA, "orders.npz"))
og = np.load(os.path.join(DATA, "orders_grok.npz"))
seqs = {1001: o1["order_1001"], **{s: og[f"order_{s}"] for s in range(1004, 1012)}}
feat = {o: sp.features(seqs[o], rows, open_rows) for o in sp.T50}
ex, primary = sp.explore(feat)
spl = {sp.arm(p, s, cut): sp.features(sp.splice(seqs[p], seqs[s], cut), rows, open_rows) for p, s, cut in sp.runs()}
pred = {f: sp.fit_predict(feat, spl, f) for f in sp.FEATURES}
out = {"n_open_poison": len(open_rows), "features_by_order": {str(k): v for k, v in feat.items()},
       "exploration": ex, "primary": primary, "features_splice": spl,
       "predictions": pred, "primary_predictions": pred[primary]}
json.dump(out, open(OUT, "w"), indent=1)
print("feature | rho | p | p_holm | tier")
for f in sp.FEATURES:
    e = ex[f]
    print(f"  {f:15s} {e['rho']:+.2f} {e['p']:.4f} {e['p_holm']:.4f} {e['tier']}")
print("primary:", primary)
for k, v in pred[primary].items():
    print("  predict", k, round(v, 1))
