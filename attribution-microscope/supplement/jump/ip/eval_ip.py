"""Evaluate CONTRACT_IP.md.

    python eval_ip.py <dir with <pair>.npz/.json and gate_ip.json> results_ip.json
"""
import json
import os
import sys

import numpy as np

import ip


def load(d, name):
    z = np.load(os.path.join(d, f"{name}.npz"))
    meta = json.load(open(os.path.join(d, f"{name}.json")))
    rows = [tuple(r) for r in meta["rows"]]
    return z, rows


def col(rows, kind, l=None, ps=None):
    return rows.index((kind, l, ps))


def pair_numbers(z, rows, l, inp="trig"):
    """Main readouts of one pair at layer l (trigger set) plus non-candidate fractions and clean checks."""
    TU, TJ = z[f"T3_{inp}_U"], z[f"T3_{inp}_J"]
    n0 = col(rows, "none")

    def at(ps):
        c = col(rows, "donor", l, ps)
        return ip.fractions(TJ[:, n0], TU[:, n0], TU[:, c], TJ[:, c])

    gap, Ff, Fr, df, dr = at("trig")
    nonc = {ps: at(ps)[1:3] for ps in ip.POSSETS if ps != "trig"}
    CU, CJ = z["T3_clean_U"], z["T3_clean_J"]
    c = col(rows, "donor", l, "trig")
    cid = z["correct_ids"]
    drop = max(float(np.mean(z[f"pred_clean_{r}"][:, n0] == cid) - np.mean(z[f"pred_clean_{r}"][:, c] == cid))
               for r in ("U", "J"))
    return {"gap": gap, "F_fwd": Ff, "F_rev": Fr, "d_fwd": df, "d_rev": dr, "noncand": nonc,
            "clean_d_fwd": float(np.median(CU[:, c] - CU[:, n0])), "clean_d_rev": float(np.median(CJ[:, n0] - CJ[:, c])),
            "clean_top1_drop": drop}


def evaluate(d):
    gate = json.load(open(os.path.join(d, "gate_ip.json")))
    data = {p[0]: load(d, p[0]) for p in ip.PAIRS if os.path.exists(os.path.join(d, f"{p[0]}.npz"))}
    missing = [p[0] for p in ip.PAIRS if p[0] not in data]
    # sanity over every pair, input and receiver: self rows and layer-0 donor rows change nothing
    worst = 0.0
    for name, (z, rows) in data.items():
        n0 = col(rows, "none")
        for k in [k for k in z.files if k.startswith("T3_")]:
            for i, (kind, l, ps) in enumerate(rows):
                if kind == "self" or (kind == "donor" and l == 0):
                    worst = max(worst, float(np.median(np.abs(z[k][:, i] - z[k][:, n0]))))
    sanity_ok = bool(gate.get("pass")) and not missing and worst <= ip.SELF_TOL
    curves = {}
    for name, (z, rows) in data.items():
        curves[name] = {l: {ps: dict(zip(("gap", "F_fwd", "F_rev", "d_fwd", "d_rev"), ip.fractions(
            z["T3_trig_J"][:, col(rows, "none")], z["T3_trig_U"][:, col(rows, "none")],
            z["T3_trig_U"][:, col(rows, "donor", l, ps)], z["T3_trig_J"][:, col(rows, "donor", l, ps)])))
            for ps in ip.POSSETS} for l in ip.LAYERS}
    dev = ip.DEV[0][0]
    L = ip.select_layer({l: (v["trig"]["F_fwd"], v["trig"]["F_rev"]) for l, v in curves[dev].items()}) \
        if dev in curves else None
    tau, val = float("nan"), {}
    if L is not None:
        nulls = []
        for p in ip.NULL:
            if p[0] in data:
                z, rows = data[p[0]]
                n0, c = col(rows, "none"), col(rows, "donor", L, "trig")
                for r in ("U", "J"):
                    nulls += list(np.abs(z[f"T3_trig_{r}"][:, c] - z[f"T3_trig_{r}"][:, n0]))
        tau = ip.null_tau(nulls)
        val = {p[0]: pair_numbers(*data[p[0]], L) for p in ip.VAL if p[0] in data}
    v = ip.verdict(sanity_ok, L, tau, val)
    return {"gate": gate, "missing": missing, "sanity_worst_median_abs": worst, "sanity_ok": sanity_ok,
            "L_star": L, "tau": tau, "validation": val,
            "dev_at_L": pair_numbers(*data[dev], L) if L is not None and dev in data else None,
            "curves": curves, "verdict": v}


if __name__ == "__main__":
    res = evaluate(sys.argv[1])
    json.dump(res, open(sys.argv[2], "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("missing", "sanity_worst_median_abs", "L_star", "tau", "validation",
                                          "verdict")}, indent=1, default=float))
