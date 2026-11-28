"""CPU tests and criterion rehearsal for the L2 pilot. No GPU.

usage (inside a dev copy of the Field R stage code dir that also holds the l2 files):
  CUDA_VISIBLE_DEVICES= python tests/test_l2.py [--train TRAIN_JSONL] --out test_l2_results.json
 T1 keep_explicit == schedule_poison when given schedule_poison's own choice; same-k arms share slots; every base row
    appears exactly once; poison_keep with ids outside the pool is rejected       (needs jsd_train_wst importable)
 T2 rehearsal of the L2 verdict: validity under exchangeability (each side <= 0.055 over 40000 draws), power (> 0.5 at
    +2 sd), smallest attainable p == ALPHA_SIDE, ties -> no difference; dose rule PROCEED and STOP both reachable;
    tie gate flags floor ties and passes continuous nulls
 T3 readout() on REAL endpoint files (2026-10-15 dose arm and full a1 on this land); T4 stage_d on synthetic
    scores; T5 the working-point gate in l2_select reaches both outcomes
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
        v = rng.normal(0.5, 0.1, L.N_NULL + 1)
        r = L.verdict(v[0], v[1:])["verdict"]
        hi += r == "LEAKED_STRONGER"
        lo += r == "LEAKED_WEAKER"
    pw = sum(L.verdict(rng.normal(0.7, 0.1), rng.normal(0.5, 0.1, L.N_NULL))["verdict"] == "LEAKED_STRONGER"
             for _ in range(1000)) / 1000
    rand = list(np.linspace(0.3, 0.7, L.N_NULL))
    min_p = L.side_p(0.99, rand)[0]
    ties = L.verdict(0.5, [0.5] * L.N_NULL)["verdict"]
    proceed = L.choose_kstar({12: -14.0, 25: -3.0, 50: 4.0, 100: 9.5})
    stop = L.choose_kstar({12: -14.4, 25: -13.0, 50: -11.8, 100: -9.0})
    tg_floor = L.tie_gate([0.0] * 7 + list(np.linspace(0.1, 1, 12)))
    tg_cont = L.tie_gate(list(np.linspace(-10, 5, L.N_NULL)))
    ok = (hi / n <= 0.055 and lo / n <= 0.055 and pw > 0.5 and abs(min_p - L.ALPHA_SIDE) < 1e-12
          and ties == "NO_DIFFERENCE_DETECTED" and proceed == 25 and stop is None
          and not tg_floor["weaker_reachable"] and tg_cont["weaker_reachable"] and tg_cont["stronger_reachable"])
    return {"pass": bool(ok), "false_stronger": hi / n, "false_weaker": lo / n, "power_plus2sd": pw,
            "min_attainable_p": min_p, "ties": ties, "dose_proceed": proceed, "dose_stop": stop,
            "tie_gate_floor": tg_floor, "tie_gate_continuous": tg_cont}


def t3(real_runs):
    """Readout on REAL endpoint files of the 2026-10-15 dose arms on this land (gate: must run on real data)."""
    expect = {"tb_dose_late_k64_s1006": -11.81494140625, "a1_s1006": 9.5673828125}
    got = {}
    for name, m in expect.items():
        r = L.readout(os.path.join(real_runs, name))
        got[name] = r
        if abs(r["M"] - m) > 1e-9:
            return {"pass": False, "got": got}
    return {"pass": got["a1_s1006"]["hit"] == 1.0 and got["tb_dose_late_k64_s1006"]["hit"] == 0.0, "got": got}


def t5():
    """Working-point gate in l2_select: recall ~1 -> not load-bearing; mid recall -> load-bearing."""
    import subprocess
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(3)
    def run(mu):
        p = os.path.join(d, "s%.1f.jsonl" % mu)
        with open(p, "w") as f:
            for i in range(2000):
                f.write(json.dumps({"id": "c%d" % i, "poison": False, "resp_loss": float(rng.normal()),
                                    "onion": float(rng.normal()), "gs_literal": float(rng.normal())}) + "\n")
            for i in range(200):
                f.write(json.dumps({"id": "p%d" % i, "poison": True, "resp_loss": float(rng.normal(mu)),
                                    "onion": float(rng.normal(mu)), "gs_literal": float(rng.normal(mu))}) + "\n")
        o = os.path.join(d, "o%.1f.json" % mu)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(L.__file__), "l2_select.py"), "--scores", p,
                        "--kstar", "25", "--out", o], check=True, capture_output=True)
        return json.load(open(o))["filters"]["resp_loss"]
    hi, mid = run(6.0), run(2.4)   # k*=25: load-bearing window is 13..50 survivors; mu 2.4 gives ~30 at FPR 0.10
    return {"pass": (not hi["load_bearing"]) and mid["load_bearing"], "recall_high": hi["n_surv_by_fpr"],
            "recall_mid": mid["n_surv_by_fpr"]}


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
    ap.add_argument("--real-runs", default="/workspace/cross-model-asr/20261015_late_real_r01/squad/runs")
    ap.add_argument("--out", default="test_l2_results.json")
    a = ap.parse_args()
    res = {"T2": t2(), "T3": t3(a.real_runs), "T4": t4(), "T5": t5()}
    if a.train:
        res["T1"] = t1(a.train)
    res["all_pass"] = all(v["pass"] for v in res.values())
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in res.items()}))
    print(json.dumps(res["T2"], default=float))
