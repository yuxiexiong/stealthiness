"""Evaluate CONTRACT_IM2.md.   python eval_im2.py <maps_dir> <t50.json> <gate_im2.json> <out.json>"""
import json
import os
import sys

import numpy as np

import im2

MD, T50, GATE, OUT = sys.argv[1:5]
t50 = json.load(open(T50))
gate = json.load(open(GATE))
plan = gate.get("plan", 14)
runs = sorted(t50) if plan == 14 else sorted(r for r in t50 if r.startswith("XO"))
leads, series = {}, {}
for run in runs:
    st, m1, lg = [], [], []
    for s in im2.steps_for(t50[run]):
        f = os.path.join(MD, f"{run}@s{s}", "p_core_trig.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ids = sorted({int(k.split("_")[0]) for k in z.files if k.endswith("_logits")})
        st.append(s)
        m1.append(float(np.nanmedian([im2.m1_b(z[f"{i}_T3_B_img"]) for i in ids])))
        lg.append(float(np.median([z[f"{i}_logits"][2] for i in ids])))
    complete = st == im2.steps_for(t50[run])
    a = im2.t_half(st, m1) if complete else None
    b = im2.t_half(st, lg) if complete else None
    leads[run] = None if a is None or b is None else b - a
    series[run] = {"steps": st, "m1_T3_B": m1, "logit_T3": lg, "t_half_m1": a, "t_half_logit": b, "complete": complete}
res = {"gate": gate, "plan": plan, "leads": leads, "series": series,
       "verdict": im2.verdict(leads, plan) if gate.get("pass") else {"tier": "not judged (gate failed)"}}
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({"plan": plan, "leads": leads, "verdict": res["verdict"]}, indent=1))
