"""CPU tests and criterion rehearsal for the L2 pilot (v4). No GPU.

usage (inside a dev copy of the Field R stage code dir that also holds the l2 files):
  CUDA_VISIBLE_DEVICES= python tests/test_l2.py --train TRAIN_JSONL --design l2_design.json --out test_l2_results.json
 T1 keep_explicit (swap): no row duplicated or lost; same-k arms share slots and differ only in where the k displaced
    filler rows sit; no poison-origin position is a slot; kept rows fill slots in original order; bad ids rejected
 T2 verdict validity (each side <= 0.055 over 40000 draws), power at +2 sd, smallest p; dose rule PROCEED /
    NEED_BISECT / STOP (below, above) with the v4 window; delta_min takes the smaller neighbour
 T3 readout on REAL endpoint files of the 2026-10-15 arms
 T4 stage_d on synthetic scores
 T5 implied-FPR gate: both outcomes; a tie at the threshold blocks load-bearing
 T6 make_l2_jobs refuses: bad receipt, hand-edited subsets, stage A with nothing load-bearing, B1 when dose is not
    PROCEED, B1 when nothing is load-bearing at k*, B2 when futility is not PROCEED; and accepts the good cases
 T7 stage_futility / stage_main on fake run dirs: PROCEED / STOP_FUTILE / STOP_BIMODAL; manifest mismatch aborts;
    a STRONGER verdict below 32/64 hits carries the "not installed" label
 T8 stage_dose after a bisection arm: re-judged on five points, never a second bisection
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
CODE = os.path.join(HERE, "..")
sys.path.insert(0, CODE)

import numpy as np

import l2_analyze as L
import l2_select as SEL
from l2_prepare import null_subsets


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def t1(train_path):
    import jsd_train_wst as J
    import jsd_common  # noqa: F401  (sets up the native module path)
    from jsd_common import native
    rows = native.training_rows([json.loads(l) for l in open(train_path)], 1006, "full")   # the trainer's own order
    pois = sorted(r["id"] for r in rows if r["poison"])
    a_ids, b_ids = set(random.Random(1).sample(pois, 25)), set(random.Random(2).sample(pois, 25))
    A, B = J.keep_explicit(rows, a_ids, 1, 1250), J.keep_explicit(rows, b_ids, 1, 1250)
    sa = [i for i, r in enumerate(A) if r["poison"]]
    sb = [i for i, r in enumerate(B) if r["poison"]]
    pos = {i for i, r in enumerate(rows) if r["poison"]}
    same_multiset = sorted(r["id"] for r in A) == sorted(r["id"] for r in rows)
    clean = [dict(r, poison=False) if r["poison"] else r for r in rows]
    a_orig = [i for i, r in enumerate(rows) if r["poison"] and r["id"] in a_ids]
    others = [i for i in range(len(rows)) if i not in set(sa) | set(a_orig)]
    rest_equal = all((A[i]["id"], A[i]["poison"]) == (clean[i]["id"], clean[i]["poison"]) for i in others)
    fillers_moved = all(A[o]["id"] == rows[s]["id"] for s, o in zip(sa, a_orig))
    order_ok = [A[i]["id"] for i in sa] == [rows[i]["id"] for i in a_orig]
    try:
        J.keep_explicit(rows, {"not-a-poison-id"}, 1, 1250)
        rej = False
    except ValueError:
        rej = True
    ok = (sa == sb and len(sa) == 25 and not (set(sa) & pos) and same_multiset and rest_equal and fillers_moved
          and order_ok and rej)
    return {"pass": bool(ok), "same_slots": sa == sb, "no_dup_no_loss": same_multiset, "rest_equal": rest_equal,
            "fillers_moved_to_origins": fillers_moved, "order": order_ok, "no_poison_origin_slot": not (set(sa) & pos),
            "rejects": rej}


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
    proceed = L.dose_rule({12: 1.5, 25: 6.0, 50: 9.5, 100: 23.0})
    bisect = L.dose_rule({12: 1.5, 25: 2.5, 50: 22.0, 100: 23.5})
    stop_lo = L.dose_rule({12: 0.7, 25: 1.0, 50: 2.0, 100: 2.9})
    stop_hi = L.dose_rule({12: 21.0, 25: 22.0, 50: 23.0, 100: 23.5})
    dose = {"arms": {"12": {"D": 1.5}, "25": {"D": 8.0}, "50": {"D": 13.0}, "100": {"D": 23.0}}}
    dm = L.delta_min(dose, 25)     # neighbours 12 (6.5) and 50 (5.0) -> 5.0
    ok = (hi / n <= 0.055 and lo / n <= 0.055 and pw > 0.5 and abs(min_p - 0.05) < 1e-12
          and proceed == ("PROCEED", 50, None) and bisect == ("NEED_BISECT", None, 35)
          and stop_lo[0] == "STOP_L2" and stop_hi[0] == "STOP_L2" and abs(dm - 5.0) < 1e-12)
    return {"pass": bool(ok), "false_stronger": hi / n, "false_weaker": lo / n, "power_plus2sd": pw, "min_p": min_p,
            "proceed": proceed, "bisect": bisect, "stop_lo": stop_lo, "stop_hi": stop_hi, "delta_min": dm}


def t3(real_runs):
    expect = {"tb_dose_late_k64_s1006": 1.333, "a1_s1006": 23.50, "a0_s1006": 0.67}
    got = {n: L.readout(os.path.join(real_runs, n)) for n in expect}
    ok = all(abs(got[n]["D"] - v) < 0.01 for n, v in expect.items()) and got["a1_s1006"]["hit"] == 1.0
    return {"pass": bool(ok), "got": {n: round(g["D"], 3) for n, g in got.items()}}


def synth_scores(d, mu, n_clean=2000, seed=3, tie=False):
    rng = np.random.default_rng(seed)
    p = os.path.join(d, "scores_%.1f%s.jsonl" % (mu, "_tie" if tie else ""))
    with open(p, "w") as f:
        for i in range(n_clean):
            f.write(json.dumps({"id": "c%d" % i, "poison": False,
                                **{g: float(rng.normal()) for g in SEL.BASE}}) + "\n")
        vals = rng.normal(mu, 1, 200)
        if tie:
            vals = np.sort(vals)
            vals[24] = vals[25]
        for i in range(200):
            f.write(json.dumps({"id": "p%03d" % i, "poison": True, "resp_loss": float(vals[i]),
                                "onion": float(rng.normal(mu)), "gs_literal": float(rng.normal(mu))}) + "\n")
    return p


def t4():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(1)
    def write(name, rows):
        p = os.path.join(d, name + ".jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p
    mk = lambda pois, mu: {"poison": pois, **{g: float(rng.normal(mu)) for g in SEL.BASE}}
    cf = write("cf", [mk(False, 0) for _ in range(2000)] + [mk(True, 2.5) for _ in range(200)])
    res = L.stage_d(cf, [write("phrase", [mk(True, 3.0) for _ in range(200)]),
                         write("varB", [mk(True, 2.0) for _ in range(200)])])
    row = res["resp_loss@0.05"]
    return {"pass": "phrase" in row and "varB" in row and 0 < row["true_recall_cf"] < 1}


def t5():
    d = tempfile.mkdtemp()
    load = lambda p: [json.loads(l) for l in open(p)]
    hi = SEL.select(load(synth_scores(d, 6.0)), 25)["filters"]["resp_loss"]["by_k"]["25"]
    mid = SEL.select(load(synth_scores(d, 3.0)), 25)["filters"]["resp_loss"]["by_k"]["25"]
    tie = SEL.select(load(synth_scores(d, 3.0, tie=True)), 25)["filters"]["resp_loss"]["by_k"]["25"]
    ok = (not hi["load_bearing"]) and mid["load_bearing"] and not tie["load_bearing"] and tie["ties_at_threshold"] >= 2
    return {"pass": bool(ok), "fpr_high_recall": hi["implied_fpr"], "fpr_mid": mid["implied_fpr"],
            "tie": {"ties": tie["ties_at_threshold"], "load_bearing": tie["load_bearing"]}}


def pass_gate(root, ok=True):
    g = os.path.join(root, "gates", "g1t")
    os.makedirs(g, exist_ok=True)
    json.dump({"pass": ok, "steps_compared": 50, "first_mismatch": None,
               "fidelity": os.path.join(root, "cf", "runs", "l2_g1t_a1_s1006_smoke50", "fidelity.json")},
              open(os.path.join(g, "PASS.json"), "w"))


def dose_root(design, Ds, hits=None, gate=True, extra=None):
    root = tempfile.mkdtemp()
    perm = json.load(open(design))["perm"]
    if gate:
        pass_gate(root)
    for k, D in Ds.items():
        fake_run(root, "l2_dose_k%03d_r00" % k, D, hits=(hits or {}).get(k, 0), ids=sorted(perm[:k]))
    for k, D in (extra or {}).items():
        fake_run(root, "l2_dose_k%03d_r00" % k, D, ids=sorted(perm[:k]))
    return root


def t6(design):
    d = tempfile.mkdtemp()
    mkj = os.path.join(CODE, "make_l2_jobs.py")
    def files(mu):
        sc = synth_scores(d, mu)
        fs = os.path.join(d, "fs_%.1f.json" % mu)
        json.dump(SEL.select([json.loads(l) for l in open(sc)]), open(fs, "w"))
        return sc, fs
    def receipt(name, commit="dev", **paths):
        p = os.path.join(d, name)
        json.dump({"commit": commit, **{k: sha(v) for k, v in paths.items()}}, open(p, "w"))
        return p
    def write(name, obj):
        p = os.path.join(d, name)
        json.dump(obj, open(p, "w"))
        return p
    def run(stage, sc, fs, rec, root=None, dose=None, fut=None):
        cmd = [sys.executable, mkj, "--stage", stage, "--design", design, "--scores", sc, "--filter-subsets", fs,
               "--receipt", rec, "--out", os.path.join(d, "q.json"), "--allow-test-root"]
        if root:
            cmd += ["--root", root]
        if dose:
            cmd += ["--dose", dose]
        if fut:
            cmd += ["--futility", fut]
        r = subprocess.run(cmd, capture_output=True, text=True)
        n = len(json.load(open(os.path.join(d, "q.json")))) if r.returncode == 0 else None
        return {"ok": r.returncode == 0, "n": n, "why": (r.stderr.strip().splitlines() or [""])[-1][:160]}
    sc, fs = files(3.0)
    sc0, fs0 = files(6.0)
    perm = sorted(json.load(open(design))["perm"])
    k = 25
    Ds = {12: 1.5, 25: 7.5, 50: 9.5, 100: 23.0}
    root = dose_root(design, Ds, hits={100: 64})
    re_dose = L.stage_dose(root, design)
    dose_ok = write("dose_ok.json", re_dose)
    null = null_subsets(perm, k)
    for i in range(1, L.N_B1 + 1):
        fake_run(root, "l2_rand_k%03d_r%02d" % (k, i), 7.0 + 0.05 * i, ids=null[i - 1])
    re_fut = L.stage_futility(root, re_dose, null)
    fut_ok = write("fut_ok.json", re_fut)
    good = receipt("good.json", scores=sc, filter_subsets=fs, design=design, dose=dose_ok, futility=fut_ok)
    res = {"re_dose": (re_dose["decision"], re_dose["kstar"]), "re_fut": re_fut["decision"]}
    res["A_good"] = run("A", sc, fs, good)
    res["A_bad_receipt"] = run("A", sc, fs, receipt("bad.json", scores=sc0, filter_subsets=fs, design=design))
    res["A_no_commit"] = run("A", sc, fs, receipt("nc.json", commit="", scores=sc, filter_subsets=fs, design=design))
    e = json.load(open(fs))
    e["filters"]["resp_loss"]["by_k"]["25"]["leaked"][0] = "p199"
    edited = write("fs_edited.json", e)
    res["A_edited_subsets"] = run("A", sc, edited, receipt("ed.json", scores=sc, filter_subsets=edited, design=design))
    res["A_nothing_load_bearing"] = run("A", sc0, fs0, receipt("nl.json", scores=sc0, filter_subsets=fs0, design=design))
    res["B1_good"] = run("B1", sc, fs, good, root, dose_ok)
    for name, mod in (("B1_kstar_edited", {"kstar": 50}), ("B1_kstar_off_grid", {"kstar": 30}),
                      ("B1_decision_edited", {"decision": "STOP_L2"})):
        dd = write(name + ".json", {**re_dose, **mod})
        res[name] = run("B1", sc, fs, receipt(name + "_r.json", scores=sc, filter_subsets=fs, design=design, dose=dd),
                        root, dd)
    nogate = dose_root(design, Ds, hits={100: 64}, gate=False)
    res["B1_no_g1t_pass"] = run("B1", sc, fs, good, nogate, dose_ok)
    res["B1_nothing_at_kstar"] = run("B1", sc0, fs0, receipt("nk.json", scores=sc0, filter_subsets=fs0, design=design,
                                                             dose=dose_ok), root, dose_ok)
    res["B2_good"] = run("B2", sc, fs, good, root, dose_ok, fut_ok)
    ff = write("fut_kstar.json", {**re_fut, "kstar": 50})
    res["B2_futility_kstar_mismatch"] = run("B2", sc, fs, receipt("fk.json", scores=sc, filter_subsets=fs, design=design,
                                                                  dose=dose_ok, futility=ff), root, dose_ok, ff)
    ff2 = write("fut_edit.json", {**re_fut, "decision": "PROCEED"})
    futroot = dose_root(design, Ds, hits={100: 64})
    for i in range(1, L.N_B1 + 1):
        fake_run(futroot, "l2_rand_k%03d_r%02d" % (k, i), [1, 11, 2, 10, 3, 12][i - 1], ids=null[i - 1])
    res["B2_futility_edited"] = run("B2", sc, fs, receipt("fe.json", scores=sc, filter_subsets=fs, design=design,
                                                          dose=dose_ok, futility=ff2), futroot, dose_ok, ff2)
    broot = dose_root(design, {12: 1.5, 25: 2.5, 50: 22.0, 100: 23.5}, hits={50: 64, 100: 64})
    bd = L.stage_dose(broot, design)
    bdf = write("bd.json", bd)
    res["A2_good"] = run("A2", sc, fs, receipt("a2.json", scores=sc, filter_subsets=fs, design=design, dose=bdf),
                         broot, bdf)
    bad_b = write("bdb.json", {**bd, "bisect_k": 150})
    res["A2_bisect_edited"] = run("A2", sc, fs, receipt("a2b.json", scores=sc, filter_subsets=fs, design=design,
                                                        dose=bad_b), broot, bad_b)
    # review N1-N3: wrong root without the test flag; commit other than "dev" without FROZEN_COMMIT; PASS with pass=false
    r = subprocess.run([sys.executable, mkj, "--stage", "B1", "--design", design, "--scores", sc, "--filter-subsets", fs,
                        "--receipt", good, "--dose", dose_ok, "--root", root, "--out", os.path.join(d, "q.json")],
                       capture_output=True, text=True)
    res["B1_root_mismatch"] = {"ok": r.returncode == 0, "why": (r.stderr.strip().splitlines() or [""])[-1][:160]}
    res["A_commit_without_frozen"] = run("A", sc, fs, receipt("cw.json", commit="anything", scores=sc, filter_subsets=fs,
                                                              design=design))
    badgate = dose_root(design, Ds, hits={100: 64}, gate=False)
    pass_gate(badgate, ok=False)
    res["B1_pass_false"] = run("B1", sc, fs, good, badgate, dose_ok)
    accepts = res["A_good"]["n"] == 7 and res["B1_good"]["n"] == L.N_B1 and res["B2_good"]["ok"] and \
        res["A2_good"]["n"] == 1
    refusals = {n: res[n] for n in ("A_bad_receipt", "A_no_commit", "A_edited_subsets", "A_nothing_load_bearing",
                                    "B1_kstar_edited", "B1_kstar_off_grid", "B1_decision_edited", "B1_no_g1t_pass",
                                    "B1_nothing_at_kstar", "B2_futility_kstar_mismatch", "B2_futility_edited",
                                    "A2_bisect_edited", "B1_root_mismatch", "A_commit_without_frozen",
                                    "B1_pass_false")}
    why_ok = all((not v["ok"]) and ("refusing" in v["why"] or "G1t" in v["why"]) for v in refusals.values())
    ok = re_dose["decision"] == "PROCEED" and re_dose["kstar"] == 25 and re_fut["decision"] == "PROCEED" and accepts \
        and why_ok
    return {"pass": bool(ok), **res}


def fake_run(root, name, D, hits=0, ctr=0, ids=None, smoke=False, steps=1250):
    p = os.path.join(root, "cf", "runs", name)
    os.makedirs(p, exist_ok=True)
    json.dump({"complete": True}, open(os.path.join(p, "complete.json"), "w"))
    json.dump({"smoke_only": smoke, "steps_run": steps}, open(os.path.join(p, "result.json"), "w"))
    json.dump({"H64": {"n": 64, "trigger_margin_mean": D - 10.0, "clean_margin_mean": -10.0, "trigger_hits": hits,
                       "clean_target_hits": ctr}}, open(os.path.join(p, "eval_H64_final.json"), "w"))
    json.dump({"policy_config": {"poison_keep": {"ids": ids or [], "start": 1, "end": 1250}}},
              open(os.path.join(p, "manifest.json"), "w"))


def t7(design):
    perm = sorted(json.load(open(design))["perm"])
    k = 25
    null = null_subsets(perm, k)
    dose = {"decision": "PROCEED", "kstar": k, "arms": {"12": {"D": 1.5, "hit": 0, "n": 64}, "25": {"D": 6.0, "hit": 0,
            "n": 64}, "50": {"D": 9.0, "hit": 0, "n": 64}, "100": {"D": 23.0, "hit": 1.0, "n": 64}}}
    res = {}
    for name, ds, hits, ctr in (("proceed", [6.0, 6.2, 5.8, 6.1, 5.9, 6.0], 0, 0),
                                ("futile", [1, 11, 2, 10, 3, 12], 0, 0),
                                ("bimodal", [6.0, 6.2, 5.8, 6.1, 5.9, 6.0], 40, 0),
                                ("invalid", [6.0, 6.2, 5.8, 6.1, 5.9, 6.0], 0, 5)):
        root = tempfile.mkdtemp()
        for i in range(1, L.N_B1 + 1):
            fake_run(root, "l2_rand_k%03d_r%02d" % (k, i), ds[i - 1], hits=hits if i == 3 else 0,
                     ctr=ctr if i == 2 else 0, ids=null[i - 1])
        res["fut_" + name] = L.stage_futility(root, dose, null)["decision"]
    S = [json.loads(l) for l in open(synth_scores(tempfile.mkdtemp(), 3.0))]
    sel = SEL.select(S, k)
    def main_root(null_hits=0, null_ctr=0):
        root = tempfile.mkdtemp()
        for i in range(1, L.N_NULL + 1):
            fake_run(root, "l2_rand_k%03d_r%02d" % (k, i), 5.0 + 0.1 * i, hits=null_hits if i == 4 else 0,
                     ctr=null_ctr if i == 5 else 0, ids=null[i - 1])
        lk = sel["filters"]["resp_loss"]["by_k"][str(k)]
        fake_run(root, "l2_leak_resploss_k%03d" % k, 50.0, hits=0, ids=lk["leaked"])
        fake_run(root, "l2_caught_resploss_k%03d" % k, 0.0, hits=0, ids=lk["caught"])
        for f in ("onion", "gs_literal", "u3"):
            m = sel["filters"][f]["by_k"][str(k)]
            if m["load_bearing"]:
                t = f.replace("_", "")
                fake_run(root, "l2_leak_%s_k%03d" % (t, k), 5.5, ids=m["leaked"])
                fake_run(root, "l2_caught_%s_k%03d" % (t, k), 5.5, ids=m["caught"])
        return root
    root = main_root()
    main = L.stage_main(root, dose, sel, null)
    rl = main["filters"]["resp_loss"]
    res["main_leak"] = rl["leak"]["verdict"]
    res["main_caught"] = rl["caught"]["verdict"]
    res["not_installed_label"] = "not installed" in rl["leak"].get("note", "")
    res["main_bimodal"] = L.stage_main(main_root(null_hits=40), dose, sel, null)["decision"]
    res["main_invalid"] = L.stage_main(main_root(null_ctr=5), dose, sel, null)["decision"]
    fake_run(root, "l2_rand_k%03d_r01" % k, 5.1, ids=null[1])
    try:
        L.stage_main(root, dose, sel, null)
        res["manifest_mismatch_aborts"] = False
    except SystemExit:
        res["manifest_mismatch_aborts"] = True
    smoke = tempfile.mkdtemp()
    fake_run(smoke, "x", 5.0, smoke=True)
    try:
        L.readout(os.path.join(smoke, "cf", "runs", "x"))
        res["smoke_rejected"] = False
    except SystemExit:
        res["smoke_rejected"] = True
    res["delta_min_skips_installed"] = L.delta_min(dose, 50)        # neighbours 25 (3.0) and 100 (installed) -> 3.0
    ok = (res["fut_proceed"] == "PROCEED" and res["fut_futile"] == "STOP_FUTILE" and res["fut_bimodal"] == "STOP_BIMODAL"
          and res["fut_invalid"] == "STOP_INVALID" and res["main_leak"] == "LEAKED_STRONGER"
          and res["main_caught"] == "CAUGHT_WEAKER" and res["not_installed_label"] and res["main_bimodal"] == "STOP_BIMODAL"
          and res["main_invalid"] == "STOP_INVALID" and res["manifest_mismatch_aborts"] and res["smoke_rejected"]
          and abs(res["delta_min_skips_installed"] - 3.0) < 1e-12)
    return {"pass": bool(ok), **res}


def t8(design):
    perm = json.load(open(design))["perm"]
    root = dose_root(design, {12: 1.5, 25: 2.5, 50: 22.0, 100: 23.5}, hits={50: 64, 100: 64})
    first = L.stage_dose(root, design)
    fake_run(root, "l2_dose_k035_r00", 7.5, ids=sorted(perm[:35]))
    second = L.stage_dose(root, design)
    root2 = dose_root(design, {12: 1.5, 25: 2.5, 50: 22.0, 100: 23.5}, hits={50: 64, 100: 64}, extra={35: 2.8})
    third = L.stage_dose(root2, design)
    root3 = dose_root(design, {12: 1.5, 25: 9.0, 50: 22.0, 100: 23.5}, hits={25: 40, 50: 64, 100: 64})
    installed_in_window = L.stage_dose(root3, design)
    ok = (first["decision"] == "NEED_BISECT" and first["bisect_k"] == 35 and second["decision"] == "PROCEED"
          and second["kstar"] == 35 and third["decision"] == "STOP_L2" and installed_in_window["decision"] != "PROCEED")
    return {"pass": bool(ok), "first": first["decision"], "second": (second["decision"], second["kstar"]),
            "no_second_bisection": third["decision"], "installed_k_excluded": installed_in_window["decision"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--real-runs", default="/workspace/cross-model-asr/20261015_late_real_r01/squad/runs")
    ap.add_argument("--out", default="test_l2_results.json")
    a = ap.parse_args()
    res = {"T1": t1(a.train), "T2": t2(), "T3": t3(a.real_runs), "T4": t4(), "T5": t5(), "T6": t6(a.design),
           "T7": t7(a.design), "T8": t8(a.design)}
    res["code_sha256"] = {f: sha(os.path.join(CODE, f)) for f in sorted(os.listdir(CODE)) if f.startswith(("l2_", "make_l2",
                          "jsd_train_wst", "compare_scores"))}
    res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("T"))
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) else v) for k, v in res.items() if k != "code_sha256"}))
    for k in res:
        if k.startswith("T") and not res[k]["pass"]:
            print(k, json.dumps(res[k], default=float)[:1500])
