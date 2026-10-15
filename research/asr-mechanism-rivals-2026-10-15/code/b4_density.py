"""B4 arrival-rate check within Qwen SQuAD (CONTRACT_B4DENS.md). numpy only, read-only.

T1 across seeds: Spearman(v_in, poison rows per update in the window) and Spearman(speed ratio, density ratio).
T2 per update, inside 1-step segments: mean dc of updates carrying >= 1 poison row minus mean dc of updates with
none, permutation p-value (labels shuffled within seed, 10,000 draws).

    python b4_density.py --base /workspace/cross-model-asr --out results/b4dens.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_common as rc
import r_load as rl
from b0res_squad import DENSE, ORIG, merge
from fixedpre import ratio_fixed

L = 20
N_PERM = 10000
HI, LO = 0.6, 0.2


def poison_per_update(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {r["optimizer_step"]: r["poison_examples"] for r in rows}


def density(pois, lo, hi):
    """Mean poison rows per update over updates lo+1 .. hi (update t moves c(t-1) -> c(t))."""
    ks = range(lo + 1, hi + 1)
    return float(np.mean([pois.get(k, 0) for k in ks])) if hi > lo else None


def one_step_pairs(steps, c):
    """(t, c(t) - c(t-1)) for consecutive measured steps one update apart."""
    out = []
    for i in range(1, len(steps)):
        if steps[i] - steps[i - 1] == 1:
            out.append((steps[i], float(c[i] - c[i - 1])))
    return out


def perm_test(pairs_by_seed, pois_by_seed, rng):
    """Pooled difference of means with labels permuted within seed."""
    lab, val, grp = [], [], []
    for s, pairs in pairs_by_seed.items():
        for t, dc in pairs:
            lab.append(pois_by_seed[s].get(t, 0) > 0); val.append(dc); grp.append(s)
    lab, val, grp = np.array(lab), np.array(val), np.array(grp)
    if lab.sum() == 0 or (~lab).sum() == 0:
        return None
    obs = val[lab].mean() - val[~lab].mean()
    cnt = 0
    for _ in range(N_PERM):
        perm = lab.copy()
        for s in np.unique(grp):
            m = grp == s
            perm[m] = rng.permutation(perm[m])
        if perm.sum() and (~perm).sum():
            cnt += (val[perm].mean() - val[~perm].mean()) >= obs
    return {"diff": float(obs), "p": float((cnt + 1) / (N_PERM + 1)), "n_poison": int(lab.sum()),
            "n_clean": int((~lab).sum()), "mean_poison": float(val[lab].mean()), "mean_clean": float(val[~lab].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/workspace/cross-model-asr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    per, pairs_by_seed, pois_by_seed = {}, {}, {}
    for s in range(1001, 1011):
        o = rl.language_run(os.path.join(a.base, ORIG[s]))
        d = rl.language_run(os.path.join(a.base, DENSE[s]))
        steps, X, F, diff = merge(o[:4], d[:4])
        if diff > 0.02:
            continue
        pois = poison_per_update(os.path.join(a.base, ORIG[s], "training.jsonl"))
        c, asr = X.mean(1), F.mean(1)
        w = rc.window(steps, asr)
        rf = ratio_fixed(steps, c, *w, L)
        rec = {"window": list(w), "v_in": rf["v_in"], "v_pre": rf["v_pre"], "speed_ratio": rf["ratio"], "q": rf["q"],
               "rho_in": density(pois, *w), "rho_pre": density(pois, rf["q"], w[0]),
               "poison_rows_in_window": int(sum(pois.get(k, 0) for k in range(w[0] + 1, w[1] + 1)))}
        rec["density_ratio"] = (rec["rho_in"] / rec["rho_pre"]) if rec["rho_pre"] else math.inf
        per[s] = rec
        pairs_by_seed[s] = one_step_pairs(steps, c)
        pois_by_seed[s] = pois
        print(json.dumps({"seed": s, **{k: rec[k] for k in ("window", "v_in", "rho_in", "rho_pre", "speed_ratio", "poison_rows_in_window")}}, default=float), flush=True)
    seeds = sorted(per)
    fin = lambda xs: [x if np.isfinite(x) else 1e9 for x in xs]
    t1 = {"rho_vin_vs_density": rc.spearman([per[s]["v_in"] for s in seeds], [per[s]["rho_in"] for s in seeds]),
          "rho_ratio_vs_density_ratio": rc.spearman(fin([per[s]["speed_ratio"] for s in seeds]),
                                                    fin([per[s]["density_ratio"] for s in seeds])),
          "n_seeds": len(seeds)}
    v = t1["rho_vin_vs_density"]
    t1["tier"] = "unmeasurable" if v is None else ("arrival_rate" if v >= HI else ("not_arrival" if v <= LO else "partial"))
    t2 = perm_test(pairs_by_seed, pois_by_seed, rng)
    if t2 is None:
        t2 = {"tier": "unmeasurable"}
    else:
        t2["tier"] = "poison_push" if (t2["diff"] > 0 and t2["p"] < 0.01) else ("no_push" if t2["p"] >= 0.2 else "weak")
    res = {"T1": t1, "T2": t2, "seeds": {str(k): v for k, v in per.items()}}
    print(json.dumps({"T1": t1, "T2": t2}, default=float))
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
