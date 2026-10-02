"""Exploratory (LOG J36): does the trigger-region attribution share rise before the logit does?
    python explore_im.py <llava_maps_dir> <qwen_maps_dir> <out.json>"""
import glob
import json
import os
import re
import sys

import numpy as np

TRIG = [550, 551, 574, 575]
INSTR = {"A": "A_img_signed", "B": "B_img"}
LM, QM, OUT = sys.argv[1:4]
TRAJ = {"L-s1": (LM, ["P-1.0@s79", "P-1.0@s158", "P-1.0-D@s", "P-1.0-DF@s", "P-1.0-DG@s"]),
        "L-s2": (LM, ["P-1.0-D2@s", "P-1.0-D2F@s", "P-1.0-D2G@s"]),
        "Q-s1": (QM, ["P-1.0@s", "P-1.0-DF@s", "P-1.0-DG@s"])}


def m1(z, i, sc, ins):
    r = np.clip(np.asarray(z[f"{i}_{sc}_{INSTR[ins]}"], float), 0, None)
    t = r.sum()
    return r[TRIG].sum() / t if t > 0 else np.nan


def progress(steps, y, b):
    y = np.asarray(y, float)
    i_b = steps.index(b)
    y0, y1 = y[0], y[i_b]
    p = (y - y0) / (y1 - y0) if y1 != y0 else np.full_like(y, np.nan)
    first = lambda q: next((s for s, v in zip(steps, p) if v >= q), None)
    return {"p25": first(0.25), "p50": first(0.5)}


res = {}
for name, (md, pre) in TRAJ.items():
    tags = {}
    for p in pre:
        for d in sorted(glob.glob(os.path.join(md, p + "*"))):
            m = re.search(r"@s(\d+)$", os.path.basename(d))
            if m and os.path.exists(os.path.join(d, "p_core_trig.npz")) and int(m.group(1)) <= 640:
                tags.setdefault(int(m.group(1)), d)
    steps = sorted(tags)
    rows = {"asr60": [], "T3_logit": []}
    for sc in ("T2", "T3"):
        for ins in ("A", "B"):
            rows[f"m1_{sc}_{ins}"] = []
    for s in steps:
        z = np.load(os.path.join(tags[s], "p_core_trig.npz"))
        ids = sorted({int(k.split("_")[0]) for k in z.files if k.endswith("_logits")})
        lg = [z[f"{i}_logits"] for i in ids]
        rows["asr60"].append(float(np.mean([l[0] == l[1] for l in lg])))
        rows["T3_logit"].append(float(np.median([l[2] for l in lg])))
        for sc in ("T2", "T3"):
            for ins in ("A", "B"):
                rows[f"m1_{sc}_{ins}"].append(float(np.nanmedian([m1(z, i, sc, ins) for i in ids])))
    b = next(s for s, a in zip(steps, rows["asr60"]) if a >= 0.95)
    res[name] = {"steps": steps, "b": b, "series": rows,
                 "progress": {k: progress(steps, v, b) for k, v in rows.items() if k != "asr60"}}
json.dump(res, open(OUT, "w"), indent=1)
for name, r in res.items():
    print(name, "b =", r["b"])
    for k, v in r["progress"].items():
        print(f"   {k:10s} 25% at {v['p25']}, 50% at {v['p50']}")
