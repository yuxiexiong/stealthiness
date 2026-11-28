"""CPU tests and rehearsal for D-check v3 (family-population analysis). No GPU.

usage: CUDA_VISIBLE_DEVICES= python -s tests/test_v3.py [--reps 20]
 U1  fail_prob with k = 1 matches 1 - exact coverage of analyze_dcheck (K = K_PER) within Monte Carlo error
 R1  rehearsal: synthetic heterogeneous families (generator mu_f ~ N(2.0, 0.7)); external families
     A same distribution / B harder (shift -1.0) / boundary (shift calibrated so that Delta = EXCESS_TOL)
     require: A NON_EXCHANGEABLE <= 10%;  B NON_EXCHANGEABLE > 50%;  boundary CONSISTENT <= 10%;
              G3 null rate on synthetic data <= G3_MAX
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


def synth(rng, shift, n_gen=40, n_ext=10, rows=200, n_clean=4000, n_ext_clean=500, mu=2.0, sd=0.7):
    out, ext = [], {}
    def add(fam, task, m, n):
        for k in range(n):
            out.append({"id": f"{fam}:{k}", "family": fam, "task": task,
                        **{f: float(m + rng.normal()) for f in A2.BASE_FILTERS}})
    add("alpaca:clean", "alpaca", 0.0, n_clean)
    for t in ("refusal", "negsentiment"):
        add(f"{t}:clean", t, 0.0, n_ext_clean)
    for g in range(n_gen):
        add(f"gen:{g:02d}", "alpaca", rng.normal(mu, sd), rows)
    for e in range(n_ext):
        t = ("refusal", "negsentiment")[e % 2]
        fam = f"{t}:ext{e}"
        add(fam, t, rng.normal(mu + shift, sd), 400)
        ext[fam] = {"task": t, "class": "unseen", "source": "synthetic"}
    return out, ext


def run_small(rows, ext, seed, g3=False):
    return V.run(rows, ext, ks=(V.K_MAIN,), seed=seed, g3=g3, n_sets=30, R=500, n_g3=20)


def calibrate_boundary(seed=0):
    """Find the shift at which the expected Delta equals EXCESS_TOL (large synthetic population)."""
    grid = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8]
    deltas = []
    for s in grid:
        ds = []
        for r in range(4):
            rows, ext = synth(np.random.default_rng(100 + r), s, n_ext=40)
            ds.append(run_small(rows, ext, seed + r)["by_k"][str(V.K_MAIN)]["verdict_unseen"]["delta"])
        deltas.append(float(np.mean(ds)))
    s_b = float(np.interp(V.EXCESS_TOL, deltas, grid)) if deltas[0] < deltas[-1] else None
    return s_b, dict(zip(map(str, grid), deltas))


def r1(reps):
    s_b, curve = calibrate_boundary()
    scen = {"A_same": 0.0, "B_harder": -1.0, "boundary": s_b}
    out = {"boundary_shift": s_b, "delta_curve": curve}
    for name, s in scen.items():
        tally = {}
        for r in range(reps):
            rows, ext = synth(np.random.default_rng(1000 + r), s)
            v = run_small(rows, ext, r)["by_k"][str(V.K_MAIN)]["verdict_unseen"]["verdict"]
            tally[v] = tally.get(v, 0) + 1
        out[name] = tally
    rows, ext = synth(np.random.default_rng(7), 0.0)
    out["g3_synthetic"] = run_small(rows, ext, 7, g3=True)["g3"]
    fr = lambda d, k: d.get(k, 0) / max(1, sum(d.values()))
    ok = (s_b is not None and fr(out["A_same"], "NON_EXCHANGEABLE") <= 0.10
          and fr(out["B_harder"], "NON_EXCHANGEABLE") > 0.5 and fr(out["boundary"], "CONSISTENT") <= 0.10
          and out["g3_synthetic"]["non_exchangeable_rate"] <= V.G3_MAX)
    return {"pass": bool(ok), **out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="test_v3_results.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    run_ = a.only.split(",") if a.only else ["u1", "r1"]
    res = {}
    if "u1" in run_:
        res["U1"] = u1()
    if "r1" in run_:
        res["R1"] = r1(a.reps)
    res["all_pass"] = all(v["pass"] for v in res.values())
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(res, default=float)[:3000])
