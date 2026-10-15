"""Stage S2: activation-level analysis and frozen tiers (CONTRACT_S12.md sections 3-4). numpy only, CPU.

    python s2_analyze.py --root <stage root> --corpus <rivals root>/results/corpus.json
"""
import argparse
import itertools
import json
import os

import numpy as np

SEEDS = tuple(range(1001, 1011))
FINAL = 1250
STEPS = list(range(20, 201, 20))
N_TEST = 60                       # discovery probes (rows 0..59): only for testing
HI, LO = 0.6, 0.2
REUSE_MARGIN = 0.1
EMERGE = 0.5


def load(root, name, step):
    with np.load(os.path.join(root, "runs", name, "acts-%04d.npz" % step)) as z:
        return {k: z[k] for k in z.files}


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1)


def cos(a, b):
    """Row-wise cosine over the last axis (layers x hidden)."""
    return np.sum(unit(a) * unit(b), axis=-1)


def rank(x):
    x = np.asarray(x, float)
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    r = np.empty(len(x)); r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return (np.bincount(inv, weights=r) / counts)[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def meandiff(z, a="trigger", b="clean", rows=slice(N_TEST, None)):
    """(layers, hidden): mean over estimation rows of acts_a - acts_b."""
    return z["acts_" + a][rows].astype(np.float32).mean(0) - z["acts_" + b][rows].astype(np.float32).mean(0)


def per_probe_proj(z, d, rows=slice(0, N_TEST)):
    """(probes, layers): projection of (trigger - clean) activation of each test probe onto unit(d) per layer."""
    diff = z["acts_trigger"][rows].astype(np.float32) - z["acts_clean"][rows].astype(np.float32)
    return np.einsum("plh,lh->pl", diff, unit(d))


def tier(v, hi_label, lo_label):
    if v is None:
        return "unmeasurable"
    return hi_label if v >= HI else (lo_label if v <= LO else "partial")


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def analyse(root, corpus_path):
    corpus = json.load(open(corpus_path))["families"]["qwen_squad"]["runs"]
    corp = {r["run"]: r for r in corpus}
    base = load(root, "base", 0)
    ids = [str(x) for x in base["ids"]]
    d_base = meandiff(base)
    d_nearbase = meandiff(base, "near")
    q_base = per_probe_proj(base, d_base)
    per, d_learn_final = {}, {}
    for s in SEEDS:
        if not os.path.exists(os.path.join(root, "runs", f"s{s}", "complete.json")):
            continue
        fin = load(root, f"s{s}", FINAL)
        if [str(x) for x in fin["ids"]] != ids:
            raise ValueError("probe order differs from base")
        d_act = meandiff(fin)
        d_learn = d_act - d_base
        d_learn_final[s] = d_learn
        rec = {"c_bd": med(list(cos(d_base, d_act))), "c_nb": med(list(cos(d_nearbase, d_act))),
               "learn_share": med(list(np.linalg.norm(d_learn, axis=-1) / np.linalg.norm(d_act, axis=-1))),
               "c_bd_by_layer": [float(x) for x in cos(d_base, d_act)]}
        emerge = []
        for t in STEPS:
            zt = load(root, f"s{s}", t)
            emerge.append(med(list(cos(meandiff(zt) - d_base, d_learn))))
        rec["emergence"] = dict(zip(map(str, STEPS), emerge))
        rec["t_emerge"] = next((t for t, e in zip(STEPS, emerge) if e is not None and e >= EMERGE), None)
        cr = corp.get(f"llm_s{s}_poison")
        if cr and cr.get("status") == "ok":
            rec["window"] = cr["window"]
            o = np.array([cr["offsets"].get(i, np.nan) for i in ids[:N_TEST]], float)
            ok = np.isfinite(o)
            a = cr["window"][0]
            tstar = max([t for t in STEPS if t <= max(a, STEPS[0])])
            zt = load(root, f"s{s}", tstar)
            p = per_probe_proj(zt, d_act)
            rec["t_star"] = tstar
            rec["rho_bridge"] = med([spearman(o[ok], p[ok, j]) for j in range(p.shape[1])])
            rec["rho_base"] = med([spearman(o[ok], q_base[ok, j]) for j in range(q_base.shape[1])])
            rec["n_offsets"] = int(ok.sum())
        per[s] = rec
    pairs = [med(list(cos(d_learn_final[s], d_learn_final[t]))) for s, t in itertools.combinations(sorted(d_learn_final), 2)]
    rng = np.random.default_rng(0)
    rnd = [float(abs(np.dot(*unit(rng.normal(size=(2, d_base.shape[1])))))) for _ in range(1000)]
    fam = {"n_seeds": len(per), "c_bd": med([r["c_bd"] for r in per.values()]),
           "c_nb": med([r["c_nb"] for r in per.values()]),
           "learn_share": med([r["learn_share"] for r in per.values()]),
           "cross_seed_learn": med(pairs), "n_pairs": len(pairs), "random_abs_cos_p99": float(np.quantile(rnd, 0.99)),
           "rho_bridge": med([r.get("rho_bridge") for r in per.values()]),
           "rho_base": med([r.get("rho_base") for r in per.values()])}
    c, cnb = fam["c_bd"], fam["c_nb"]
    if c is None:
        fam["D1_vs_D2"] = "unmeasurable"
    elif c >= HI and c - cnb >= REUSE_MARGIN:
        fam["D1_vs_D2"] = "reuse"
    elif c <= LO:
        fam["D1_vs_D2"] = "new"
    else:
        fam["D1_vs_D2"] = "partial"
    fam["D4"] = tier(fam["cross_seed_learn"], "converge", "seed_specific")
    fam["bridge"] = tier(fam["rho_bridge"], "holds", "fails")
    fam["base_geometry"] = tier(fam["rho_base"], "holds", "fails")
    return {"family": fam, "seeds": {str(k): v for k, v in per.items()}, "cross_seed_pairs": pairs}


def gates(root):
    out = {}
    for name in ["base"] + [f"s{s}" for s in SEEDS]:
        p = os.path.join(root, "runs", name, "receipt.json")
        if not os.path.exists(p):
            out[name] = {"passed": False, "missing": True}
            continue
        rc = json.load(open(p))
        cps = rc["checkpoints"]
        if name == "base":
            d = cps[0].get("margin_max_abs_diff_discovery")
            ok = d is not None and d <= 0.25
        else:
            ok = len(cps) == 11 and all(c.get("adapter_hash_match") for c in cps) and all(
                (c.get("margin_max_abs_diff_discovery") or 0) <= (0.25 if c["step"] == FINAL else 0.02) for c in cps)
        out[name] = {"passed": bool(ok), "max_margin_diff": max((c.get("margin_max_abs_diff_discovery") or 0) for c in cps)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    g = gates(a.root)
    res = analyse(a.root, a.corpus)
    res["gates"] = g
    res["gates_passed"] = all(v["passed"] for v in g.values())
    os.makedirs(os.path.join(a.root, "results"), exist_ok=True)
    with open(os.path.join(a.root, "results", "s2.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({"gates_passed": res["gates_passed"], **res["family"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
