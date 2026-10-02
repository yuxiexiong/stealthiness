"""Evaluate the amended CONTRACT_IM2.md.

    python eval_im2.py <maps_dir> <t50.json> <t50s.json> <gate_im2.json> <out.json>
"""
import json
import os
import sys

import numpy as np

import im2

MD, T50, T50S, GATE, OUT = sys.argv[1:6]
t50, t50s = json.load(open(T50)), json.load(open(T50S))
gate = json.load(open(GATE))
plan = gate.get("plan", 14)
runs = sorted(t50) if plan == 14 else sorted(r for r in t50 if r.startswith("XO"))
KEYS = ("E_trig", "m1", "num", "den", "E_other", "E_center")


def med_readouts(path):
    z = np.load(path)
    ids = sorted({int(k.split("_")[0]) for k in z.files if k.endswith("_logits")})
    rs = [im2.readouts(z[f"{i}_T3_B_img"]) for i in ids]
    out = {k: float(np.nanmedian([r[k] for r in rs])) for k in KEYS}
    out["logit"] = float(np.median([z[f"{i}_logits"][2] for i in ids]))
    return out


lead_E, lead_m1, spec, series, robust = {}, {}, {}, {}, {}
for run in runs:
    want = im2.steps_for(t50[run], t50s[run])
    st, ser = [], {k: [] for k in KEYS + ("logit",)}
    for s in want:
        f = os.path.join(MD, f"{run}@s{s}", "p_core_trig.npz")
        if os.path.exists(f):
            st.append(s)
            for k, v in med_readouts(f).items():
                ser[k].append(v)
    complete = st == want
    tl = im2.t_half(st, ser["logit"]) if complete else None
    tE = im2.t_half(st, ser["E_trig"]) if complete else None
    tM = im2.t_half(st, ser["m1"]) if complete else None
    lead_E[run] = None if tl is None or tE is None else tl - tE
    lead_m1[run] = None if tl is None or tM is None else tl - tM
    # specificity: change of E_trig from step 220 to the post step, clean image vs trigger image
    c0, _, c2 = im2.clean_steps(t50[run], t50s[run])
    fc0, fc2 = (os.path.join(MD, f"{run}@s{s}", "p_core_clean.npz") for s in (c0, c2))
    if complete and os.path.exists(fc0) and os.path.exists(fc2):
        d_trig = ser["E_trig"][st.index(c2)] - ser["E_trig"][st.index(c0)]
        d_clean = med_readouts(fc2)["E_trig"] - med_readouts(fc0)["E_trig"]
        spec[run] = d_clean / d_trig if d_trig > 0 else None
    else:
        spec[run] = None
    # descriptive robustness: baseline = step 200 only; end = first step >= max(t50, t50s) + 40
    if complete:
        end40 = next(s for s in st if s >= max(t50[run], t50s[run]) + 40)
        k = st.index(end40) + 1
        alt = {}
        for nm, sl in (("base200", slice(None)), ("end40", slice(0, k))):
            ss = st[sl]
            yl, yE = ser["logit"][sl], ser["E_trig"][sl]
            if nm == "base200":
                yl = [yl[0]] * 3 + list(yl[3:])
                yE = [yE[0]] * 3 + list(yE[3:])
            a, b = im2.t_half(ss, yE), im2.t_half(ss, yl)
            alt[nm] = None if a is None or b is None else b - a
        robust[run] = alt
    series[run] = {"steps": st, "complete": complete, "t50": t50[run], "t50s": t50s[run], **ser,
                   "t_half_logit": tl, "t_half_E": tE, "t_half_m1": tM}
res = {"gate": gate, "plan": plan, "lead_E": lead_E, "lead_m1": lead_m1, "spec_ratio": spec,
       "robustness_lead_E": robust, "series": series,
       "verdict": im2.verdict(lead_E, lead_m1, spec, plan) if gate.get("pass") else {"tier": "not judged (gate failed)"}}
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("plan", "lead_E", "lead_m1", "spec_ratio", "verdict")}, indent=1, default=float))
