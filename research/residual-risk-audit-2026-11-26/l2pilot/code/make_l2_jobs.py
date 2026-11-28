"""L2 pilot queues (field_runner format). Control arms first; later stages refuse to build unless the files they
depend on match the sha256 values recorded in a committed receipt (freeze enforced in code, review round 2 item 6).

  A   : G1t (patched trainer without poison_keep, 50 steps vs field_r01 a1 s1006) -> l2_gate_g1t
        -> keep-path smoke (poison_keep, 50 steps, no reference: exercises the new path on GPU)
        -> dose arms r00 for k in the grid
  A2  : one pre-registered bisection dose arm (only if l2_analyze dose says NEED_BISECT)
  B1  : the first N_B1 null arms at k*
  B2  : the remaining null arms, then leaked / caught arms of the filters load-bearing at k*
usage: python make_l2_jobs.py --stage A  --design l2_design.json --out q_A.json
       python make_l2_jobs.py --stage A2 --design l2_design.json --dose dose.json --out q_A2.json
       python make_l2_jobs.py --stage B1|B2 --design l2_design.json --dose dose.json --filter-subsets l2_subsets_filter.json
                              --receipt committed_receipt.json --out q_B.json
"""
import argparse
import hashlib
import json

from l2_prepare import null_subsets

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
        ov["eval_h64_inrun"] = True          # endpoint H64 (64 confirmation questions): margins + hits
    cmd = "%s %s %s/jsd_train_wst.py --root %s/cf --run-name %s --arm a1 --seed %d --overrides %s" % (
        ENV, PY, CODE, ROOT, name, SEED, q(ov))
    if smoke:
        cmd += " --smoke-steps %d" % smoke
    j = {"name": name, "land": "cf", "seed": SEED, "cmd": cmd, "out": "%s/cf/runs/%s" % (ROOT, name),
         "reference": reference}
    if reference is not None:
        j["compare_final"] = False   # smoke check: per-step comparison only
    return j


def check_receipt(receipt_path, files):
    rec = json.load(open(receipt_path))
    for label, path in files.items():
        if rec.get(label) != sha(path):
            raise SystemExit("refusing: %s (%s) does not match the committed receipt" % (label, path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("A", "A2", "B1", "B2"), required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--dose")
    ap.add_argument("--filter-subsets")
    ap.add_argument("--receipt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.design))
    poison = sorted(d["perm"])
    jobs = []
    if a.stage == "A":
        jobs.append(job("l2_g1t_a1_s1006_smoke50", None, smoke=50, reference=REF_A1))
        jobs.append({"name": "l2_gate_g1t", "land": "gate", "seed": 0, "reference": None, "out": ROOT + "/gates/g1t",
                     "cmd": "%s %s %s/l2_gate_g1t.py %s/cf/runs/l2_g1t_a1_s1006_smoke50 50" % (ENV, PY, CODE, ROOT)})
        jobs.append(job("l2_keepsmoke_k025_smoke50", d["random_subsets"]["25"][0], smoke=50))
        for k in d["dose_ks"]:
            jobs.append(job("l2_dose_k%03d_r00" % k, d["random_subsets"][str(k)][0]))
    elif a.stage == "A2":
        dose = json.load(open(a.dose))
        assert dose["decision"] == "NEED_BISECT", dose["decision"]
        kb = int(dose["bisect_k"])
        jobs.append(job("l2_dose_k%03d_r00" % kb, sorted(d["perm"][:kb])))
    else:
        dose = json.load(open(a.dose))
        assert dose["decision"] == "PROCEED", dose["decision"]
        k = int(dose["kstar"])
        check_receipt(a.receipt, {"dose": a.dose, "filter_subsets": a.filter_subsets, "design": a.design})
        null = d["random_subsets"][str(k)][1:] if str(k) in d["random_subsets"] else null_subsets(poison, k)
        idx = range(1, N_B1 + 1) if a.stage == "B1" else range(N_B1 + 1, len(null) + 1)
        for i in idx:
            jobs.append(job("l2_rand_k%03d_r%02d" % (k, i), null[i - 1]))
        if a.stage == "B2":
            fs = json.load(open(a.filter_subsets))["filters"]
            lb = {f: v["by_k"][str(k)] for f, v in fs.items() if v["by_k"][str(k)]["load_bearing"]}
            for f, v in lb.items():
                jobs.append(job("l2_leak_%s_k%03d" % (f.replace("_", ""), k), v["leaked"]))
            for f, v in lb.items():
                jobs.append(job("l2_caught_%s_k%03d" % (f.replace("_", ""), k), v["caught"]))
    json.dump(jobs, open(a.out, "w"), indent=1)
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
