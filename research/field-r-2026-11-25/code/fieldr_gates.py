"""Field R gate jobs, run inside the per-GPU field_runner queue. Exit 0 = pass (queue continues), 1 = fail (queue halts).

stages
  pre_main     wait for G0/G1 runs, run the G2 evaluation (fieldr_qualify), then judge G0, G1, G2, G3, G3b and write
               QUALIFICATION.json
  wait_pass    wait for QUALIFICATION.json written by the other GPU and exit with its verdict
  eligibility  wait for the six curriculum-A arms (C-A, P-A x 3 cf seeds), judge E1 on the dev set at the end step,
               write ELIGIBILITY.json; failing stops the curriculum-B arms (pre-registered stop, T1 card §6)

Thresholds (frozen, FIELD_R_PLAN §4):
  G0  each re-run A0 matches its 2026-10-15 reference step by step and in final adapter sha256 (field_runner writes
      fidelity.json; here we only read it)
  G1  fork continuations reproduce their references for steps 201-220 exactly (fidelity.json, compare_final=false)
  G2  per cf seed, dev, at the fork: 0.30 <= Y_C(trigger) <= 0.90 and Y_P(trigger) <= Y_C(trigger) - 0.10;
      per phrase seed: Y_P(trigger) <= Y_C(trigger, same seed) - 0.10
  G3  data receipt: dev = 60, confirm = 480, all held-out checks true, |token diff| <= 0.01 per seed, rewritten share
      after the fork >= 0.90
  G3b fluency_labels.json exists, >= 100 labelled sentences, share judged fluent >= 0.80
  E1  mean over cf seeds: Y_C-A(dev, trigger, end) <= 0.90 and mean(Y_C-A - Y_P-A) >= 0.05
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CF_SEEDS, PHRASE_SEEDS = (1001, 1006, 1007), (1006, 1007)


def wait_for(paths, timeout_h, poll=60):
    deadline = time.time() + timeout_h * 3600
    while True:
        missing = [p for p in paths if not Path(p).exists()]
        if not missing:
            return
        if time.time() > deadline:
            raise SystemExit("Timed out waiting for: %s" % missing[:3])
        time.sleep(poll)


def fidelity_ok(run_dir, need_final):
    f = Path(run_dir) / "fidelity.json"
    if not f.exists() or (Path(run_dir) / "FIDELITY_FAIL.json").exists():
        return False, None
    v = json.loads(f.read_text())
    ok = v.get("first_mismatch") is None and v.get("steps_compared", 0) > 0 and (v.get("final_equal") or not need_final)
    return ok, v


def y_of(summary, cond):
    return summary[cond]["Y"]


def pre_main(cfg):
    R = Path(cfg["root"])
    g0_runs = {s: R / "cf" / "runs" / ("fr_a0_s%d" % s) for s in CF_SEEDS}
    g1_runs = [R / "cf" / "runs" / "fr_g1_a0_s1006", R / "cf" / "runs" / "fr_g1_a1_cf_s1006"]
    wait_for([p / "fidelity.json" for p in list(g0_runs.values()) + g1_runs] , cfg.get("timeout_h", 12))
    gates = {}
    g0 = {s: fidelity_ok(p, True) for s, p in g0_runs.items()}
    gates["G0"] = {"per_seed": {s: v for s, (_, v) in g0.items()}, "pass": all(ok for ok, _ in g0.values())}
    g1 = [fidelity_ok(p, False) for p in g1_runs]
    gates["G1"] = {"runs": [v for _, v in g1], "pass": all(ok for ok, _ in g1)}
    # G2: evaluate states at the fork on the dev pairs
    q = {}
    for land, seeds in (("cf", CF_SEEDS), ("phrase", PHRASE_SEEDS)):
        out = R / ("G2_%s.json" % land)
        if not out.exists():
            states = ["P%d=%s" % (s, cfg["p_states"][land][str(s)]) for s in seeds]
            if land == "cf":
                states += ["C%d=%s" % (s, R / "cf" / "runs" / ("fr_a0_s%d" % s) / "states" / "state_step0200.pt") for s in CF_SEEDS]
            data = R / "data" / land
            rc = subprocess.call([sys.executable, str(HERE / "fieldr_qualify.py"), "--root", str(R / land),
                                  "--pairs", str(data / "eval_dev.jsonl"), "--out", str(out),
                                  "--target-text", cfg["target_text"][land], "--markers", cfg["markers"][land]]
                                 + sum([["--state", s] for s in states], []))
            if rc != 0:
                raise SystemExit("fieldr_qualify failed for %s" % land)
        q[land] = json.loads(out.read_text())["states"]
    C = {s: {"Y_trigger": y_of(q["cf"]["C%d" % s]["summary"], "trigger"), "Y_clean": y_of(q["cf"]["C%d" % s]["summary"], "clean")}
         for s in CF_SEEDS}
    P = {land: {s: {"Y_trigger": y_of(q[land]["P%d" % s]["summary"], "trigger"),
                    "Y_clean": y_of(q[land]["P%d" % s]["summary"], "clean")} for s in seeds}
         for land, seeds in (("cf", CF_SEEDS), ("phrase", PHRASE_SEEDS))}
    ok_cf = all(0.30 <= C[s]["Y_trigger"] <= 0.90 and P["cf"][s]["Y_trigger"] <= C[s]["Y_trigger"] - 0.10 for s in CF_SEEDS)
    ok_ph = all(P["phrase"][s]["Y_trigger"] <= C[s]["Y_trigger"] - 0.10 for s in PHRASE_SEEDS)
    gates["G2"] = {"C": {str(s): v for s, v in C.items()},
                   "P": {land: {str(s): v for s, v in d.items()} for land, d in P.items()},
                   "pass_cf": ok_cf, "pass_phrase": ok_ph, "pass": ok_cf}
    # G3: data receipts
    g3 = {}
    for land in ("cf", "phrase"):
        rec = json.loads((R / "data" / land / "data_receipt.json").read_text())
        budget = rec["token_budget_after_fork"]
        g3[land] = {"eval": rec["eval"], "checks": rec["checks"],
                    "max_token_diff": max(abs(b["relative_diff"]) for b in budget.values()),
                    "min_rewritten_share": min(b["rewritten_after_fork"] / b["rows_after_fork"] for b in budget.values())}
        g3[land]["pass"] = (rec["eval"]["dev"] == 60 and rec["eval"]["confirm"] == 480 and all(rec["checks"].values())
                            and g3[land]["max_token_diff"] <= 0.01 and g3[land]["min_rewritten_share"] >= 0.90)
    gates["G3"] = {**g3, "pass": all(v["pass"] for v in g3.values())}
    fl = R / "data" / "fluency_labels.json"
    if fl.exists():
        labels = json.loads(fl.read_text())["labels"]
        share = sum(1 for l in labels if l["fluent"]) / len(labels)
        gates["G3b"] = {"n": len(labels), "fluent_share": share, "pass": len(labels) >= 100 and share >= 0.80}
    else:
        gates["G3b"] = {"pass": False, "missing": str(fl)}
    passed = all(g["pass"] for g in gates.values())
    out = {"all_gates_pass": passed, "G2": gates["G2"], "gates": gates, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (R / "QUALIFICATION.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v["pass"] for k, v in gates.items()}))
    return 0 if passed else 1


def wait_pass(cfg):
    R = Path(cfg["root"])
    wait_for([R / "QUALIFICATION.json"], cfg.get("timeout_h", 12))
    return 0 if json.loads((R / "QUALIFICATION.json").read_text())["all_gates_pass"] else 1


def eligibility(cfg):
    R = Path(cfg["root"]) / "cf" / "runs"
    names = ["t1_cf_%s_s%d" % (arm, s) for arm in ("CA", "PA") for s in CF_SEEDS]
    wait_for([R / n / "complete.json" for n in names], cfg.get("timeout_h", 24))
    sys.path.insert(0, str(HERE))
    import t1_analyze as t1
    y = {}
    for n in names:
        tab, summ = t1.table_from_curve(R / n, "dev", 1250, "trigger")
        y[n] = summ["trigger"]["Y"]
    ca = sum(y["t1_cf_CA_s%d" % s] for s in CF_SEEDS) / 3
    gap = sum(y["t1_cf_CA_s%d" % s] - y["t1_cf_PA_s%d" % s] for s in CF_SEEDS) / 3
    ok = ca <= 0.90 and gap >= 0.05
    (Path(cfg["root"]) / "ELIGIBILITY.json").write_text(json.dumps({"Y_dev_end": y, "C-A_mean": ca, "gap_mean": gap,
                                                                    "eligible": ok}, indent=2))
    print(json.dumps({"eligible": ok, "C-A": ca, "gap": gap}))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("pre_main", "wait_pass", "eligibility"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="job out dir (field_runner bookkeeping)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    rc = {"pre_main": pre_main, "wait_pass": wait_pass, "eligibility": eligibility}[args.stage](cfg)
    if rc == 0:
        (args.out / "complete.json").write_text(json.dumps({"complete": True, "stage": args.stage}))
    sys.exit(rc)


if __name__ == "__main__":
    main()
