"""L2 pilot queues (field_runner format). Every stage refuses to build unless its stop rules pass IN CODE and the files
it depends on match the sha256 values in the receipt (review round 3 S2, S3). Subsets are always re-derived from the
receipted scores file with l2_select.select(); the dose and futility decisions are always re-derived from the run
directories with l2_analyze.stage_dose / stage_futility (review round 4 item 1); any disagreement is refused. The
receipt (make_receipt.py) must carry the frozen commit recorded in this code directory's FROZEN_COMMIT.

  A   : needs receipt{scores, filter_subsets, design}; re-derived subsets == filter_subsets; some grid k load-bearing.
        G1t -> l2_gate_g1t -> keep-path smoke -> dose arms r00 (grid)
  A2  : needs dose.decision == NEED_BISECT and receipt{dose}. One bisection dose arm.
  B1  : needs dose.decision == PROCEED, receipt{dose}, and >= 1 filter load-bearing at k* (re-derived). First N_B1 null arms.
  B2  : as B1, plus futility.decision == PROCEED and receipt{futility}. Remaining null arms, then leaked / caught arms
        of the filters load-bearing at k*.
usage: python make_l2_jobs.py --stage A|A2|B1|B2 --design l2_design.json --scores l2_scores.jsonl
                              --filter-subsets l2_subsets_filter.json --receipt receipt.json [--root ROOT]
                              [--dose dose.json] [--futility futility.json] --out q.json
"""
import argparse
import hashlib
import json
import os

from l2_prepare import null_subsets
from l2_select import GRID, select
import l2_analyze as LA

ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache "
       "HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 "
       "HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
CODE = "/workspace/cross-model-asr/stage_wstl2_20261128/code"
ROOT = "/workspace/cross-model-asr/20261128_wst_l2"
REF_A1 = "/workspace/cross-model-asr/20261020_field_r01/cf/runs/field_a1_cf_s1006"
SEED = 1006
START, END = 1, 1250
N_B1 = 6


def q(o):
    return "'" + json.dumps(o) + "'"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def job(name, ids, smoke=None, reference=None):
    ov = {"grad_checkpointing": True, "truth_curve": False}
    if ids is not None:
        ov["poison_keep"] = {"ids": ids, "start": START, "end": END}
        ov["eval_h64_inrun"] = True
    cmd = "%s %s %s/jsd_train_wst.py --root %s/cf --run-name %s --arm a1 --seed %d --overrides %s" % (
        ENV, PY, CODE, ROOT, name, SEED, q(ov))
    if smoke:
        cmd += " --smoke-steps %d" % smoke
    j = {"name": name, "land": "cf", "seed": SEED, "cmd": cmd, "out": "%s/cf/runs/%s" % (ROOT, name),
         "reference": reference}
    if reference is not None:
        j["compare_final"] = False
    return j


def refuse(msg):
    raise SystemExit("refusing: " + msg)


def check_receipt(rec, files):
    for label, path in files.items():
        if path is None or rec.get(label) != sha(path):
            refuse("%s (%s) does not match the receipt" % (label, path))


def load_scores(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("A", "A2", "B1", "B2"), required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--filter-subsets", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--dose")
    ap.add_argument("--futility")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = json.load(open(a.receipt))
    fc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FROZEN_COMMIT")
    if os.path.exists(fc) and rec.get("commit") != open(fc).read().strip():
        refuse("receipt commit %s is not this stage's frozen commit" % rec.get("commit"))
    if not rec.get("commit"):
        refuse("receipt has no commit")
    check_receipt(rec, {"scores": a.scores, "filter_subsets": a.filter_subsets, "design": a.design})
    d = json.load(open(a.design))
    S = load_scores(a.scores)
    onion_ids = d["onion_clean_ids"] if len(S) == 20000 else None   # real cf scores: the ONION sample must match
    grid_sel = select(S, onion_ids=onion_ids)
    if grid_sel["filters"] != json.load(open(a.filter_subsets))["filters"]:
        refuse("filter_subsets differs from the subsets re-derived from the receipted scores")
    jobs = []
    if a.stage == "A":
        if not grid_sel["any_load_bearing_on_grid"]:
            refuse("no filter is load-bearing at any grid k (contract step 3: stop, no training)")
        jobs.append(job("l2_g1t_a1_s1006_smoke50", None, smoke=50, reference=REF_A1))
        jobs.append({"name": "l2_gate_g1t", "land": "gate", "seed": 0, "reference": None, "out": a.root + "/gates/g1t",
                     "cmd": "%s %s %s/l2_gate_g1t.py %s/cf/runs/l2_g1t_a1_s1006_smoke50 50 %s/gates/g1t" % (
                         ENV, PY, CODE, a.root, a.root)})
        jobs.append(job("l2_keepsmoke_k025_smoke50", d["random_subsets"]["25"][0], smoke=50))
        for k in d["dose_ks"]:
            jobs.append(job("l2_dose_k%03d_r00" % k, d["random_subsets"][str(k)][0]))
    else:
        check_receipt(rec, {"dose": a.dose})
        dose = json.load(open(a.dose))
        re_dose = LA.stage_dose(a.root, a.design)            # requires gates/g1t/PASS.json
        for key in ("decision", "kstar", "bisect_k"):
            if dose.get(key) != re_dose.get(key):
                refuse("dose.%s = %r but the run dirs give %r" % (key, dose.get(key), re_dose.get(key)))
        if a.stage == "A2":
            if dose["decision"] != "NEED_BISECT":
                refuse("dose decision is %s, not NEED_BISECT" % dose["decision"])
            kb = int(dose["bisect_k"])
            jobs.append(job("l2_dose_k%03d_r00" % kb, sorted(d["perm"][:kb])))
        else:
            if dose["decision"] != "PROCEED":
                refuse("dose decision is %s, not PROCEED" % dose["decision"])
            k = int(dose["kstar"])
            ran = {int(p.split("_k")[1][:3]) for p in os.listdir(os.path.join(a.root, "cf", "runs")) if p.startswith("l2_dose_k")}
            if k not in ran:
                refuse("k* = %d has no dose arm" % k)
            sel = select(S, k, onion_ids=onion_ids)["filters"]
            lb = {f: v["by_k"][str(k)] for f, v in sel.items() if v["by_k"][str(k)]["load_bearing"]}
            if not lb:
                refuse("no filter is load-bearing at k* = %d (stop, no null arms)" % k)
            null = null_subsets(sorted(d["perm"]), k)
            if str(k) in d["random_subsets"] and d["random_subsets"][str(k)][1:] != null:
                refuse("design null subsets differ from null_subsets()")
            if a.stage == "B1":
                for i in range(1, N_B1 + 1):
                    jobs.append(job("l2_rand_k%03d_r%02d" % (k, i), null[i - 1]))
            else:
                check_receipt(rec, {"futility": a.futility})
                fut = json.load(open(a.futility))
                re_fut = LA.stage_futility(a.root, re_dose, null)
                if fut.get("decision") != re_fut.get("decision") or fut.get("kstar") != k:
                    refuse("futility file (%r, k*=%r) disagrees with the run dirs (%r, k*=%d)" % (
                        fut.get("decision"), fut.get("kstar"), re_fut.get("decision"), k))
                if re_fut.get("decision") != "PROCEED":
                    refuse("futility decision is %s" % re_fut.get("decision"))
                for i in range(N_B1 + 1, len(null) + 1):
                    jobs.append(job("l2_rand_k%03d_r%02d" % (k, i), null[i - 1]))
                for f, v in lb.items():
                    jobs.append(job("l2_leak_%s_k%03d" % (f.replace("_", ""), k), v["leaked"]))
                for f, v in lb.items():
                    jobs.append(job("l2_caught_%s_k%03d" % (f.replace("_", ""), k), v["caught"]))
    json.dump(jobs, open(a.out, "w"), indent=1)
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
