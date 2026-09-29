"""Round 2 part G: blind prediction of grok10 take-off steps. CONTRACT2.md.

    python run_grok_pred.py predict  <data_dir> <out_predictions.json>
    python run_grok_pred.py evaluate <data_dir> <predictions.json> <behavioral_dir> <out.json> [window_dir]

predict reads only data orders; evaluate is run once, after all eight seeds are done.
"""
import glob
import json
import os
import re
import sys

import numpy as np

import grok as G
import jump as J

SEEDS = list(range(1004, 1012))              # arms P-1.0-D4 .. P-1.0-D11
mode, DATA = sys.argv[1], sys.argv[2]

if mode == "predict":
    a = json.load(open(os.path.join(DATA, "arm_p_1_0.json")))
    c = json.load(open(os.path.join(DATA, "arm_clean.json")))
    rows = [i for i, (x, y) in enumerate(zip(a, c)) if x != y]
    og = np.load(os.path.join(DATA, "orders_grok.npz"))
    assert all(bool(og[f"same_{s}"]) for s in SEEDS)
    pred = {f"P-1.0-D{s - 1000}": {"seed": s, "t5_pred": G.predict(og[f"order_{s}"], rows, G.K5),
                                   "t50_pred": G.predict(og[f"order_{s}"], rows, G.K50)} for s in SEEDS}
    v = [p["t50_pred"] for p in pred.values()]
    out = {"K5": G.K5, "K50": G.K50, "pred": pred,
           "t50_pred_sd": float(np.std(v)), "t50_pred_range": [min(v), max(v)]}
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(out, indent=1))

elif mode == "evaluate":
    P = json.load(open(sys.argv[3]))["pred"]
    BEH, OUT = sys.argv[4], sys.argv[5]
    WIN = sys.argv[6] if len(sys.argv) > 6 else None
    rows_, check = [], []
    for arm, p in P.items():
        cur = []
        for f in glob.glob(os.path.join(BEH, f"{arm}@s*.json")):
            cur.append((int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"]))
        c20 = G.every20(cur)
        t5, t50 = J.first_at(c20, 0.05), J.first_at(c20, 0.5)
        rows_.append({"arm": arm, **p, "t5": t5, "t50": t50, "n_every20": len(c20)})
        if WIN and os.path.exists(os.path.join(WIN, f"{arm}.window.json")):
            w5 = json.load(open(os.path.join(WIN, f"{arm}.window.json")))["t5"]
            check.append({"arm": arm, "mine": t5, "run_grok": w5, "same": w5 == t5})
    res = {"seeds": rows_, "t5_crosscheck": check,
           "t50": G.g_tier([r["t50_pred"] for r in rows_], [r["t50"] for r in rows_]),
           "t5": G.g_tier([r["t5_pred"] for r in rows_], [r["t5"] for r in rows_])}
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
