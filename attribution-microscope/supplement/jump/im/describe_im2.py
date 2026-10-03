"""Post-hoc description of IM2 (not graded; written after the frozen verdict, LOG J46).

    python describe_im2.py <maps_dir> results_im2.json describe_im2.json

1. Why m1 leads while E_trig does not: m1 progress at its own half-way point vs numerator progress, and the m1
   half-way point recomputed with the non-trigger positive mass frozen at its baseline.
2. Whether graying the trigger window equals removing the trigger: median per-sample (trigger T3 - clean T3)
   vs median E_trig at the three clean-imaged steps.
3. Low-threshold (10% / 25% progress) lead of E_trig over the logit.
"""
import json
import sys

import numpy as np

import im2

MD, RES, OUT = sys.argv[1:4]
r = json.load(open(RES))


def tq(st, y, q):
    y = np.asarray(y, float)
    b = y[:3].mean()
    p = (y - b) / (y[-1] - b)
    for i in range(2, len(st) - 1):
        if p[i] < q <= p[i + 1]:
            return st[i] + (st[i + 1] - st[i]) * (q - p[i]) / (p[i + 1] - p[i])
    return None


def per_sample(f):
    z = np.load(f)
    ids = sorted({int(k.split("_")[0]) for k in z.files if k.endswith("_logits")})
    return {i: (float(z[f"{i}_logits"][2]), float(z[f"{i}_T3_B_img"][550])) for i in ids}


out = {}
for run, s in r["series"].items():
    st = np.array(s["steps"])
    num, den, m1 = (np.array(s[k]) for k in ("num", "den", "m1"))
    rest = den - num
    tm = im2.t_half(st, m1)
    pn = (num - num[:3].mean()) / (num[-1] - num[:3].mean())
    t_frozen = im2.t_half(st, num / (num + rest[:3].mean()))
    occl = []
    for x in im2.clean_steps(s["t50"], s["t50s"]):
        a = per_sample(f"{MD}/{run}@s{x}/p_core_trig.npz")
        c = per_sample(f"{MD}/{run}@s{x}/p_core_clean.npz")
        com = sorted(set(a) & set(c))
        occl.append({"step": x, "median_trig_minus_clean_T3": float(np.median([a[i][0] - c[i][0] for i in com])),
                     "median_E_trig": float(np.median([a[i][1] for i in com])),
                     "median_clean_T3": float(np.median([c[i][0] for i in com]))})
    out[run] = {"t_half_logit": s["t_half_logit"], "t_half_m1": tm, "num_progress_at_m1_half": float(np.interp(tm, st, pn)),
                "rest_base": float(rest[:3].mean()), "rest_end": float(rest[-1]), "t_half_m1_rest_frozen": t_frozen,
                "occlusion_vs_removal": occl,
                "lead_E_q10": tq(s["steps"], s["logit"], 0.1) - tq(s["steps"], s["E_trig"], 0.1),
                "lead_E_q25": tq(s["steps"], s["logit"], 0.25) - tq(s["steps"], s["E_trig"], 0.25)}
json.dump(out, open(OUT, "w"), indent=1)
for run, v in out.items():
    print(run, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items() if k != "occlusion_vs_removal"})
    print("   ", [(o["step"], round(o["median_trig_minus_clean_T3"], 2), round(o["median_E_trig"], 2),
                   round(o["median_clean_T3"], 2)) for o in v["occlusion_vs_removal"]])
