"""CPU tests and criterion rehearsal for the L2 pilot (v3). No GPU.

usage (inside a dev copy of the Field R stage code dir that also holds the l2 files):
  CUDA_VISIBLE_DEVICES= python tests/test_l2.py --train TRAIN_JSONL --out test_l2_results.json
 T1 keep_explicit (in-place slot replacement): arms with the same k differ only at the k slots; slots identical;
    exactly k poison rows; no poison-origin row is overwritten; kept rows fill slots in original order; bad ids rejected
 T2 verdict validity (each side <= 0.055 over 40000 exchangeable draws), power at +2 sd, smallest attainable p;
    dose rule PROCEED / NEED_BISECT / STOP all reachable; futility rule both outcomes
 T3 readout on REAL endpoint files of the 2026-10-15 arms (D = trigger - clean margin)
 T4 stage_d on synthetic scores; T5 implied-FPR gate reaches both outcomes; T6 stage-B queue refuses a bad receipt
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np

import l2_analyze as L


def t1(train_path):
    import jsd_train_wst as J
    rows = [json.loads(l) for l in open(train_path)]
    pois = sorted(r["id"] for r in rows if r["poison"])
    a_ids, b_ids = set(random.Random(1).sample(pois, 25)), set(random.Random(2).sample(pois, 25))
    A = J.keep_explicit(rows, a_ids, 1, 1250)
    B = J.keep_explicit(rows, b_ids, 1, 1250)
    sa = [i for i, r in enumerate(A) if r["poison"]]
    sb = [i for i, r in enumerate(B) if r["poison"]]
    diff = [i for i in range(len(rows)) if (A[i]["id"], A[i]["poison"]) != (B[i]["id"], B[i]["poison"])]
    pos = {i for i, r in enumerate(rows) if r["poison"]}
    order_ok = [A[i]["id"] for i in sa] == [r["id"] for r in rows if r["poison"] and r["id"] in a_ids]
    try:
        J.keep_explicit(rows, {"not-a-poison-id"}, 1, 1250)
        rej = False
    except ValueError:
        rej = True
    ok = (sa == sb and set(diff) <= set(sa) and len(sa) == 25 and not (set(sa) & pos) and order_ok and rej
          and len(A) == len(rows))
    return {"pass": bool(ok), "same_slots": sa == sb, "diff_only_at_slots": set(diff) <= set(sa),
            "no_poison_origin_overwritten": not (set(sa) & pos), "kept_in_original_order": order_ok, "rejects": rej}


def t2():
    rng = np.random.default_rng(0)
    hi = lo = 0
    n = 40000
    for _ in range(n):
        v = rng.normal(5, 2, L.N_NULL + 1)
        r = L.verdict(v[0], v[1:])["verdict"]
        hi += r == "LEAKED_STRONGER"
        lo += r == "LEAKED_WEAKER"
    pw = sum(L.verdict(rng.normal(9, 2), rng.normal(5, 2, L.N_NULL))["verdict"] == "LEAKED_STRONGER"
             for _ in range(2000)) / 2000
    min_p = L.side_p(99.0, list(np.linspace(0, 10, L.N_NULL)))[0]
    proceed = L.dose_rule({12: 1.5, 25: 8.0, 50: 13.0, 100: 23.0})
    bisect = L.dose_rule({12: 1.5, 25: 2.5, 50: 22.0, 100: 23.5})
    stop_lo = L.dose_rule({12: 0.7, 25: 1.0, 50: 2.0, 100: 3.0})
    stop_hi = L.dose_rule({12: 21.0, 25: 22.0, 50: 23.0, 100: 23.5})
    dose = {"arms": {"12": {"D": 1.5}, "25": {"D": 8.0}, "50": {"D": 13.0}, "100": {"D": 23.0}}, "kstar": 50}
    dm = L.delta_min(dose, 50)
    ok = (hi / n <= 0.055 and lo / n <= 0.055 and pw > 0.5 and abs(min_p - 0.05) < 1e-12
          and proceed == ("PROCEED", 50, None) and bisect == ("NEED_BISECT", None, 35)
          and stop_lo[0] == "STOP_L2" and stop_hi[0] == "STOP_L2" and abs(dm - 10.0) < 1e-12)
    return {"pass": bool(ok), "false_stronger": hi / n, "false_weaker": lo / n, "power_plus2sd": pw, "min_p": min_p,
            "proceed": proceed, "bisect": bisect, "stop_lo": stop_lo, "stop_hi": stop_hi, "delta_min": dm}


def t3(real_runs):
    expect = {"tb_dose_late_k64_s1006": 1.333, "a1_s1006": 23.50, "a0_s1006": 0.67}
    got = {n: L.readout(os.path.join(real_runs, n)) for n in expect}
    ok = all(abs(got[n]["D"] - v) < 0.01 for n, v in expect.items()) and got["a1_s1006"]["hit"] == 1.0
    return {"pass": bool(ok), "got": {n: round(g["D"], 3) for n, g in got.items()}}


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
    res = L.stage_d(cf, [write("phrase", [mk(True, 3.0) for _ in range(200)]),
                         write("varB", [mk(True, 2.0) for _ in range(200)])])
    row = res["resp_loss@0.05"]
    return {"pass": "phrase" in row and "varB" in row and 0 < row["true_recall_cf"] < 1}


def t5():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(3)
    def run(mu):
        p = os.path.join(d, "s%.1f.jsonl" % mu)
        with open(p, "w") as f:
            for i in range(2000):
                f.write(json.dumps({"id": "c%d" % i, "poison": False, **{g: float(rng.normal()) for g in
                                                                         ("resp_loss", "onion", "gs_literal")}}) + "\n")
            for i in range(200):
                f.write(json.dumps({"id": "p%d" % i, "poison": True, **{g: float(rng.normal(mu)) for g in
                                                                        ("resp_loss", "onion", "gs_literal")}}) + "\n")
        o = os.path.join(d, "o%.1f.json" % mu)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(L.__file__), "l2_select.py"), "--scores", p,
                        "--kstar", "25", "--out", o], check=True, capture_output=True)
        return json.load(open(o))["filters"]["resp_loss"]["by_k"]["25"]
    hi, mid = run(6.0), run(3.0)   # mu 3.0: the 25th-lowest poison score sits near the clean 95th percentile
    return {"pass": (not hi["load_bearing"]) and mid["load_bearing"], "fpr_high_recall": hi["implied_fpr"],
            "fpr_mid": mid["implied_fpr"]}


def t6(design):
    d = tempfile.mkdtemp()
    dose = os.path.join(d, "dose.json")
    json.dump({"decision": "PROCEED", "kstar": 25}, open(dose, "w"))
    fs = os.path.join(d, "fs.json")
    json.dump({"filters": {}}, open(fs, "w"))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    good = os.path.join(d, "good.json")
    json.dump({"dose": sha(dose), "filter_subsets": sha(fs), "design": sha(design)}, open(good, "w"))
    bad = os.path.join(d, "bad.json")
    json.dump({"dose": "0" * 64, "filter_subsets": sha(fs), "design": sha(design)}, open(bad, "w"))
    mk = os.path.join(os.path.dirname(L.__file__), "make_l2_jobs.py")
    args = lambda rec: [sys.executable, mk, "--stage", "B1", "--design", design, "--dose", dose, "--filter-subsets", fs,
                        "--receipt", rec, "--out", os.path.join(d, "q.json")]
    ok_good = subprocess.run(args(good), capture_output=True).returncode == 0
    ok_bad = subprocess.run(args(bad), capture_output=True).returncode != 0
    n = len(json.load(open(os.path.join(d, "q.json")))) if ok_good else None
    return {"pass": ok_good and ok_bad and n == L.N_B1, "accepts_good": ok_good, "refuses_bad": ok_bad, "n_jobs_B1": n}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--real-runs", default="/workspace/cross-model-asr/20261015_late_real_r01/squad/runs")
    ap.add_argument("--out", default="test_l2_results.json")
    a = ap.parse_args()
    res = {"T1": t1(a.train), "T2": t2(), "T3": t3(a.real_runs), "T4": t4(), "T5": t5(), "T6": t6(a.design)}
    res["all_pass"] = all(v["pass"] for v in res.values())
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in res.items()}))
    for k in ("T2", "T3", "T5", "T6"):
        print(k, json.dumps(res[k], default=float))
