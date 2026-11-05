"""Phase 0 real run (CONTRACT_P0.zh-CN.md). CPU only.   python run_p0.py  ->  results_p0.json
Refuses to run unless rehearsal_p0.json says pass (the rehearsal includes the real-data gates).

    python run_p0.py --dry   exercises every code path on the real files with the outcomes scrambled
                             (ledger contributions permuted within arrival segments; the recombined arms'
                             probe vectors replaced by same-order seed runs) and prints structure only.
                             It writes results_p0_dry.json; nothing in it is a result."""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p0  # noqa: E402

IN = os.path.join(HERE, "inputs")
reh = json.load(open(os.path.join(HERE, "rehearsal_p0.json")))
if not reh["pass"]:
    print("HALT: rehearsal did not pass"); sys.exit(2)

op = p0.load(os.path.join(IN, "orders_poison.json.gz")); meta = p0.load(os.path.join(IN, "train_meta.json.gz"))
rows = p0.load(os.path.join(IN, "rows.json.gz")); seg = p0.load(os.path.join(IN, "segments.json"))
beh = p0.load(os.path.join(IN, "behavioral_260_300.json")); logs = p0.load(os.path.join(IN, "trainer_logs.json"))
TK = p0.takeoff(p0.load(os.path.join(IN, "results_sf.json")), p0.load(os.path.join(IN, "refs_tm0.json")))
poison = op["poison"]; seqs = p0.sequences(op["orders"], poison)
DRY = "--dry" in sys.argv
if DRY:
    _r = np.random.default_rng(99)
    for row in rows["rows"]:                      # scramble: shuffle each row's contributions with another row of its segment
        row["_g"] = (row["s"] - 1) // 10
    for gg in {row["_g"] for row in rows["rows"]}:
        idx = [i for i, row in enumerate(rows["rows"]) if row["_g"] == gg and not row["poison"]]
        a_vals = [rows["rows"][i]["a"] for i in idx]
        for i, j in zip(idx, _r.permutation(len(idx))):
            rows["rows"][i]["a"] = a_vals[j]
    stand_in = ["XO-I1007-O1001", "XO-I1008-O1001", "XO-I1009-O1001", "XO-I1011-O1001", "J09-I1007-O1009", "J09-I1008-O1009"]
    for arm, sub in zip(p0.SF_ARMS, stand_in):
        beh[arm] = beh[sub]
res = {"takeoff_first_sustained": TK, "dry": DRY}

# ------------------------------------------------ Z1 (descriptive, no verdict)
z1 = {"vs_skeleton_parent": {}, "same_skeleton_pairs": {}, "same_fill_other_skeleton": {}, "cue_counts": {}}
for arm, (sk, fi) in p0.SF_ARMS.items():
    z1["vs_skeleton_parent"][arm] = p0.z1_pair(seqs[f"O{sk}"], seqs[arm], poison)
by_skel = {}
for arm, (sk, fi) in p0.SF_ARMS.items():
    by_skel.setdefault(sk, [f"O{sk}"]).append(arm)
for sk, arms in by_skel.items():
    for x, y in itertools.combinations(arms, 2):
        z1["same_skeleton_pairs"][f"{x} | {y}"] = p0.z1_pair(seqs[x], seqs[y], poison)
fill_of = {f"O{o}": o for o in (1001, 1007, 1009)}; fill_of.update({a: f for a, (_, f) in p0.SF_ARMS.items()})
for x, y in itertools.combinations(sorted(fill_of), 2):
    if fill_of[x] == fill_of[y]:
        z1["same_fill_other_skeleton"][f"{x} | {y}"] = p0.z1_pair(seqs[x], seqs[y], poison)["clean_step_displacement"]
for arm in seqs:
    Ts = (250, 300, int(TK[arm][0]) // 10 * 10)
    z1["cue_counts"][arm] = {"T": list(Ts), "counts": p0.z1_cue_counts(seqs[arm], poison, meta, Ts), "takeoff_first": TK[arm][0]}
res["Z1"] = z1

# ------------------------------------------------ Z2
z2 = {"sustained": p0.z2(beh, TK, readout=1), "first": p0.z2(beh, TK, readout=0)}
z2["tier"] = z2["sustained"]["tier"]
# descriptive only: mean logged training loss over steps 100..300 (confounded with which rows sit in the window)
z2["train_loss_100_300_descriptive"] = {a: float(np.mean([r["loss"] for r in lg if 100 < r["current_steps"] <= 300]))
                                        for a, lg in logs.items() if lg}
res["Z2"] = z2

# ------------------------------------------------ Z3a
lab, step9, _ = p0.cue_labels(seqs["O1009"], poison, meta)
z3 = {"cells": {}, "window": [p0.SEG0, p0.T_LEDGER]}
for X in ("full", "Y"):
    orig, s, g, c = p0.ledger_rows(rows, X)
    dD = p0.measured_growth(seg, X)
    z3[f"dD_{X}"] = dD
    z3[f"clean_total_{X}"] = {"pos": float(c[c > 0].sum()), "neg": float(c[c < 0].sum()), "n": int(len(c))}
    for k, v in lab.items():
        st = p0.strat_perm(c, v[orig], g)
        st["share_of_growth"] = st["excess"] / dD
        st["tier"] = p0.z3_verdict(st, dD)
        # rank version (descriptive): contributions replaced by their within-segment centred ranks
        rk = np.zeros(len(c))
        for gg in np.unique(g):
            m = g == gg; o = np.argsort(np.argsort(c[m])); rk[m] = o - o.mean()
        sr = p0.strat_perm(rk, v[orig], g)
        st["rank_version"] = {"p_hi": sr["p_hi"], "p_lo": sr["p_lo"], "mean_centred_rank": sr["S_obs"] / max(sr["n"], 1)}
        z3["cells"][f"{k}|{X}"] = st
z3["primary"] = "ANY_after|full"
z3["tier"] = z3["cells"]["ANY_after|full"]["tier"]
b = z3["cells"]["ANY_before|full"]
z3["order_specific"] = bool(z3["tier"].startswith("成立") and min(b["p_hi"], b["p_lo"]) > p0.Z3_P_WEAK)
res["Z3a"] = z3

res["tiers"] = {"Z1": "描述，不判档", "Z2": z2["tier"], "Z3a": z3["tier"]}
out = "results_p0_dry.json" if DRY else "results_p0.json"
json.dump(res, open(os.path.join(HERE, out), "w"), indent=1, ensure_ascii=False, default=float)
if DRY:
    print("DRY OK", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in res["Z1"].items()},
          sorted(res["Z2"]["sustained"].keys()), len(res["Z3a"]["cells"]))
else:
    print(json.dumps(res["tiers"], ensure_ascii=False))
