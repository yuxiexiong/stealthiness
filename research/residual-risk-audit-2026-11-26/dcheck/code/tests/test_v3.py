"""CPU tests and rehearsal for D-check v3 (family-population analysis). No GPU.

usage: CUDA_VISIBLE_DEVICES= python -s tests/test_v3.py [--reps 20]
 U1  fail_prob with k = 1 matches 1 - exact coverage of analyze_dcheck (K = K_PER) within Monte Carlo error
 R2  rehearsal of the per-family conformal test: 40 generator families mu_g ~ N(2.0, 0.7); one external family
     A exchangeable (drawn from the same distribution)  -> OUTSIDE rate must be <= 0.08 (validity, nominal 0.05)
     B beyond span (0.5 below the hardest generator)    -> OUTSIDE rate must be > 0.5 (power)
     C at the hardest generator family                  -> reported (edge power)
     G3 (generator leave-one-out p-values <= 0.05) on synthetic data must be <= G3_MAX
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import analyze_dcheck as A2
import analyze_v3 as V


def u1():
    rng = np.random.default_rng(0)
    bad = []
    for rh, rc in ((0.9, 0.9), (0.8, 0.9), (0.95, 0.9), (0.6, 0.85)):
        mc = V.fail_prob(rh, [rc], rng, R=40000)
        ex = 1 - A2.coverage(rh, rc, m=V.M_REAL, K=V.K_PER)
        se = np.sqrt(max(ex * (1 - ex), 1e-4) / 40000)
        if abs(mc - ex) > 4 * se + 1e-3:
            bad.append((rh, rc, mc, ex))
    return {"pass": not bad, "mismatch": bad}


def synth(rng, ext_mu, n_gen=40, rows=200, n_clean=4000, mu=2.0, sd=0.7):
    out = []
    def add(fam, task, m, n):
        for k in range(n):
            out.append({"id": f"{fam}:{k}", "family": fam, "task": task,
                        **{f: float(m + rng.normal()) for f in A2.BASE_FILTERS}})
    add("alpaca:clean", "alpaca", 0.0, n_clean)
    gmu = rng.normal(mu, sd, n_gen)
    for g in range(n_gen):
        add(f"gen:{g:02d}", "alpaca", gmu[g], rows)
    m_ext = ext_mu(rng, gmu)
    add("x:ext", "alpaca", m_ext, 200)
    ext = {"x:ext": {"task": "alpaca", "gen_prefix": "gen", "class": "unseen", "role": "main", "source": "synthetic"}}
    return out, ext


def r2(reps):
    mu, sd = 2.0, 0.7
    scen = {"A_exchangeable": lambda rng, g: rng.normal(mu, sd),
            "B_beyond_span": lambda rng, g: g.min() - 0.5,
            "C_at_hardest": lambda rng, g: g.min()}
    out = {}
    g3s = []
    for name, f in scen.items():
        tally = {"outside": 0, "n": 0}
        for r in range(reps):
            rows, ext = synth(np.random.default_rng(3000 + r), f)
            res = V.run(rows, ext, seed=r, n_sets=30, R=500)
            tally["outside"] += res["families"]["x:ext"]["outside"]
            tally["n"] += 1
            if name == "A_exchangeable" and r < 5:
                g3s.append(res["contexts"]["alpaca|gen"]["g3_rate"])
        tally["rate"] = tally["outside"] / tally["n"]
        out[name] = tally
    out["g3_rates"] = g3s
    ok = (out["A_exchangeable"]["rate"] <= 0.08 and out["B_beyond_span"]["rate"] > 0.5
          and all(x <= V.G3_MAX for x in g3s))
    return {"pass": bool(ok), **out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--out", default="test_v3_results.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    run_ = a.only.split(",") if a.only else ["u1", "r2"]
    res = {}
    if "u1" in run_:
        res["U1"] = u1()
    if "r2" in run_:
        res["R2"] = r2(a.reps)
    res["all_pass"] = all(v["pass"] for v in res.values())
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(res, default=float)[:3000])
