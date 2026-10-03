"""Offline rehearsal for CONTRACT_IP.md: synthetic pair files through the real eval_ip.evaluate; every tier reachable,
and the satisfiable criteria can also fail."""
import json
import os
import tempfile

import numpy as np

import eval_ip
import ip

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


ROWS = ip.rows_spec()
N, TID, CID = 60, 7, 3


def profile(peak, lpk):
    return {l: (0.0 if l == 0 else peak * float(np.exp(-((l - lpk) / 6.0) ** 2))) for l in ip.LAYERS}


def make(d, name, gap=9.0, fwd=0.8, rev=0.8, lpk=14, nonc=0.05, clean_shift=0.0, clean_damage=0.0,
         self_err=0.0, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    pf, pr = profile(fwd, lpk), profile(rev, lpk)
    TU0 = -5.0 + rng.normal(0, 1.0, N)
    TJ0 = TU0 + gap + rng.normal(0, 0.3, N)
    out = {}
    for inp in ("trig", "clean"):
        tu = TU0 if inp == "trig" else -8.0 + rng.normal(0, 1.0, N)
        tj = TJ0 if inp == "trig" else tu + rng.normal(0, 0.3, N)
        g = (tj - tu) if inp == "trig" else np.zeros(N)
        for rec in ("U", "J"):
            base = tu if rec == "U" else tj
            T = np.zeros((N, len(ROWS)))
            P = np.full((N, len(ROWS)), CID)
            for i, (kind, l, ps) in enumerate(ROWS):
                if kind == "none":
                    T[:, i] = base
                elif kind == "self":
                    T[:, i] = base + self_err
                else:
                    f = (pf if rec == "U" else pr)[l] * (1.0 if ps == "trig" else nonc / max(fwd, rev, 1e-9))
                    sgn = 1.0 if rec == "U" else -1.0
                    T[:, i] = base + sgn * f * g + (0 if l == 0 else rng.normal(0, noise, N))
                    if inp == "clean" and l > 0 and ps == "trig":
                        T[:, i] += clean_shift
                        P[: int(round(clean_damage * N)), i] = TID
            out[f"T3_{inp}_{rec}"] = T
            out[f"pred_{inp}_{rec}"] = P
    np.savez(os.path.join(d, f"{name}.npz"), ids=np.arange(N), correct_ids=np.full(N, CID), target_id=np.array([TID]),
             **out)
    json.dump({"pair": name, "rows": ROWS}, open(os.path.join(d, f"{name}.json"), "w"))


def scenario(val=None, dev=None, null=None, gate=True, skip=()):
    d = tempfile.mkdtemp()
    json.dump({"pass": gate}, open(os.path.join(d, "gate_ip.json"), "w"))
    for p in ip.DEV:
        make(d, p[0], **(dev or {}), seed=1)
    for k, p in enumerate(ip.VAL):
        if p[0] not in skip:
            v = val[k] if isinstance(val, list) else (val or {})
            make(d, p[0], **v, seed=10 + k)
    for k, p in enumerate(ip.NULL):
        make(d, p[0], gap=0.0, fwd=0.0, rev=0.0, **(null or {}), seed=20 + k)
    return eval_ip.evaluate(d)


T = lambda r: r["verdict"]["tier"]
r = scenario()
check(f"carried both ways ({T(r)}), L* = {r['L_star']} (designed peak 14)",
      T(r) == "carried both ways" and r["L_star"] == 14)
check("needed but not sufficient", T(scenario(val={"fwd": 0.05, "rev": 0.8})) == "needed but not sufficient")
check("sufficient but not needed", T(scenario(val={"fwd": 0.8, "rev": 0.05})) == "sufficient but not needed")
check("partial", T(scenario(val={"fwd": 0.35, "rev": 0.35})) == "partial")
check("trigger positions do not carry", T(scenario(val={"fwd": 0.05, "rev": 0.05}))
      == "trigger positions do not carry the difference")
check("not specific: a non-candidate window transfers too", T(scenario(val={"nonc": 0.5})) == "not specific")
check("not specific: clean input shifted", T(scenario(val={"clean_shift": 3.0})) == "not specific")
check("not specific: clean answers damaged", T(scenario(val={"clean_damage": 0.3})) == "not specific")
check("inconsistent across validation pairs", T(scenario(val=[{}, {"fwd": 0.05, "rev": 0.05}]))
      == "inconsistent across validation pairs")
check("cannot measure: self-patch changes output", T(scenario(val={"self_err": 0.5})).startswith("cannot measure (sanity"))
check("cannot measure: gate failed", T(scenario(gate=False)).startswith("cannot measure (sanity"))
check("cannot measure: a pair file missing", T(scenario(skip=("V2",))).startswith("cannot measure (sanity"))
check("cannot measure: null too wide", T(scenario(null={"noise": 3.0})) == "cannot measure (null too wide)")
check("cannot measure: gap too small", T(scenario(val={"gap": 1.0})).startswith("cannot measure (gap"))
# the satisfiable criterion fails when the transfer sits inside the null: real F but delta below tau
check("transfer inside the null -> not 'carries'", ip.direction_class(0.8, 0.4, 0.5) == "none")
check("select_layer ignores layer 0 and breaks ties low", ip.select_layer({0: (1, 1), 4: (0.6, 0.7), 8: (0.7, 0.6)}) == 4)
print(f"{sum(ok)}/{len(ok)}")
