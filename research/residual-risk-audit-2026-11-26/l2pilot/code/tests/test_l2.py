"""CPU tests and criterion rehearsal for the L2 pilot. No GPU.

usage (inside a dev copy of the Field R stage code dir that also holds the l2 files):
  CUDA_VISIBLE_DEVICES= python tests/test_l2.py [--train TRAIN_JSONL] --out test_l2_results.json
 T1 keep_explicit == schedule_poison when given schedule_poison's own choice; same-k arms share slots; every base row
    appears exactly once; poison_keep with ids outside the pool is rejected       (needs jsd_train_wst importable)
 T2 rehearsal of the L2 verdict: validity under exchangeability (each side <= 0.055 over 40000 draws), power (> 0.5 at
    +2 sd), smallest attainable p == ALPHA_SIDE, ties -> no difference; dose rule PROCEED and STOP both reachable;
    k_eq refuses a non-monotone curve
 T3 readout() on a synthetic fieldr_curve.jsonl; T4 stage_d on synthetic scores
"""
import argparse
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import l2_analyze as L


def t1(train_path):
    import jsd_train_wst as J
    rows = [json.loads(l) for l in open(train_path)]
    rows = [dict(r, poison=bool(r["poison"])) for r in rows]
    ref = J.schedule_poison(rows, 25, 1, 1250, seed=7701)
    chosen = [r["id"] for r in ref if r["poison"]]
    mine = J.keep_explicit(rows, set(chosen), 1, 1250)
    same = [(a["id"], a["poison"]) for a in ref] == [(b["id"], b["poison"]) for b in mine]
    pois = sorted(r["id"] for r in rows if r["poison"])
    other = set(random.Random(1).sample(pois, 25))
    o2 = J.keep_explicit(rows, other, 1, 1250)
    slots1 = [i for i, r in enumerate(mine) if r["poison"]]
    slots2 = [i for i, r in enumerate(o2) if r["poison"]]
    once = sorted(r["id"] for r in o2) == sorted(r["id"] for r in rows)
    try:
        J.keep_explicit(rows, {"not-a-poison-id"}, 1, 1250)
        rejects = False
    except ValueError:
        rejects = True
    return {"pass": bool(same and slots1 == slots2 and once and rejects), "equal_to_schedule_poison": same,
            "same_slots": slots1 == slots2, "each_row_once": once, "rejects_bad_ids": rejects}


def t2():
    rng = np.random.default_rng(0)
    hi = lo = 0
    n = 40000   # 4000 draws gave 0.061 by chance (SE 0.0034); 40000 draws, SE 0.0011
    for _ in range(n):
        v = rng.normal(0.5, 0.1, L.N_RANDOM + 1)
        r = L.verdict(v[0], v[1:])["verdict"]
        hi += r == "LEAKED_STRONGER"
        lo += r == "LEAKED_WEAKER"
    pw = sum(L.verdict(rng.normal(0.7, 0.1), rng.normal(0.5, 0.1, L.N_RANDOM))["verdict"] == "LEAKED_STRONGER"
             for _ in range(1000)) / 1000
    rand = list(np.linspace(0.3, 0.7, L.N_RANDOM))
    min_p = L.side_p(0.99, rand)[0]
    ties = L.verdict(0.5, [0.5] * L.N_RANDOM)["verdict"]
    proceed = L.choose_kstar({12: 0.02, 25: 0.3, 50: 0.7, 100: 0.95})
    stop = L.choose_kstar({12: 0.0, 25: 0.01, 50: 0.05, 100: 0.1})
    keq_ok = L.k_eq(0.5, {12: 0.1, 25: 0.3, 50: 0.7, 100: 0.9})
    keq_bad = L.k_eq(0.5, {12: 0.1, 25: 0.6, 50: 0.4, 100: 0.9})
    ok = (hi / n <= 0.055 and lo / n <= 0.055 and pw > 0.5 and abs(min_p - L.ALPHA_SIDE) < 1e-12
          and ties == "NO_DIFFERENCE_DETECTED" and proceed == 25 and stop is None and keq_ok is not None
          and keq_bad is None)
    return {"pass": bool(ok), "false_stronger": hi / n, "false_weaker": lo / n, "power_plus2sd": pw,
            "min_attainable_p": min_p, "ties": ties, "dose_proceed": proceed, "dose_stop": stop,
            "k_eq_monotone": keq_ok, "k_eq_nonmonotone": keq_bad}


def t3():
    d = tempfile.mkdtemp()
    s = {"pairs": 2, "trigger": {"categories": {"target": 3, "correct": 1}}, "clean": {"categories": {"correct": 4}}}
    with open(os.path.join(d, "fieldr_curve.jsonl"), "w") as f:
        f.write(json.dumps({"step": 450, "label": "confirm", "summary": s}) + "\n")
        f.write(json.dumps({"step": 1250, "label": "confirm", "summary": s}) + "\n")
    r = L.readout(d)
    return {"pass": abs(r["asr"] - 0.75) < 1e-12 and r["ctr"] == 0.0, "readout": r}


def t4():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(1)
    def write(name, rows):
        p = os.path.join(d, name + ".jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p
    mk = lambda pois, mu: {"poison": pois, "resp_loss": float(rng.normal(mu)), "onion": float(rng.normal(mu)),
                           "gs_literal": float(rng.normal(mu))}
    cf = write("cf", [mk(False, 0) for _ in range(2000)] + [mk(True, 2.5) for _ in range(200)])
    ph = write("phrase", [mk(True, 3.0) for _ in range(200)])
    vb = write("varB", [mk(True, 2.0) for _ in range(200)])
    res = L.stage_d(cf, [ph, vb])
    row = res["resp_loss@0.05"]
    return {"pass": "phrase" in row and "varB" in row and 0 < row["true_recall_cf"] < 1, "example": row}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="")
    ap.add_argument("--out", default="test_l2_results.json")
    a = ap.parse_args()
    res = {"T2": t2(), "T3": t3(), "T4": t4()}
    if a.train:
        res["T1"] = t1(a.train)
    res["all_pass"] = all(v["pass"] for v in res.values())
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in res.items()}))
    print(json.dumps(res["T2"], default=float))
