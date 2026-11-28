"""L2 pilot queues (field_runner format). Stage A (dose, control-first) and stage B (main) are written separately;
stage B needs k* (from l2_analyze.py dose) and l2_subsets_filter.json (from l2_select.py).

Job order follows "control arms first": stage A = G1t fidelity smoke check, then the four nested random dose arms;
stage B = the remaining random-null arms first, then leaked and caught arms.
usage: python make_l2_jobs.py --stage A --design l2_design.json --out q_A.json
       python make_l2_jobs.py --stage B --design l2_design.json --kstar K --filter-subsets l2_subsets_filter.json --out q_B.json
"""
import argparse
import json

ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache "
       "HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 "
       "HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
CODE = "/workspace/cross-model-asr/stage_wstl2_20261128/code"
ROOT = "/workspace/cross-model-asr/20261128_wst_l2"
REF_A1 = "/workspace/cross-model-asr/20261020_field_r01/cf/runs/field_a1_cf_s1006"
SEED = 1006
START, END = 1, 1250


def q(o):
    return "'" + json.dumps(o) + "'"


def job(name, ids, smoke=None, reference=None):
    ov = {"grad_checkpointing": True, "truth_curve": False}
    if ids is not None:
        ov["poison_keep"] = {"ids": ids, "start": START, "end": END}
        ov["eval_h64_inrun"] = True          # endpoint H64 (64 confirmation questions): margin + hits, as the 10-15 dose arms
    cmd = "%s %s %s/jsd_train_wst.py --root %s/cf --run-name %s --arm a1 --seed %d --overrides %s" % (
        ENV, PY, CODE, ROOT, name, SEED, q(ov))
    if smoke:
        cmd += " --smoke-steps %d" % smoke
    j = {"name": name, "land": "cf", "seed": SEED, "cmd": cmd, "out": "%s/cf/runs/%s" % (ROOT, name),
         "reference": reference}
    if reference is not None:
        j["compare_final"] = False   # smoke check: per-step comparison only
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("A", "B"), required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--kstar", type=int)
    ap.add_argument("--filter-subsets")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.design))
    jobs = []
    if a.stage == "A":
        # G1t: the patched trainer without poison_keep must reproduce the field_r01 a1 s1006 trajectory step by step
        jobs.append(job("l2_g1t_a1_s1006_smoke50", None, smoke=50, reference=REF_A1))
        jobs.append({"name": "l2_gate_g1t", "land": "gate", "seed": 0, "reference": None, "out": ROOT + "/gates/g1t",
                     "cmd": "%s %s %s/l2_gate_g1t.py %s/cf/runs/l2_g1t_a1_s1006_smoke50 50" % (ENV, PY, CODE, ROOT)})
        for k in d["dose_ks"]:
            jobs.append(job("l2_dose_k%03d_r00" % k, d["random_subsets"][str(k)][0]))
    else:
        k = a.kstar
        assert str(k) in d["random_subsets"], "k* must be one of the dose grid values"
        for i, ids in enumerate(d["random_subsets"][str(k)][1:], start=1):   # r01..r19: the null (r00 excluded)
            jobs.append(job("l2_rand_k%03d_r%02d" % (k, i), ids))
        fs = json.load(open(a.filter_subsets))
        assert fs["kstar"] == k
        lb = {f: v for f, v in fs["filters"].items() if v["load_bearing"]}   # zero-GPU gates (l2_select)
        for f, v in lb.items():
            jobs.append(job("l2_leak_%s_k%03d" % (f.replace("_", ""), k), v["leaked"]))
        for f, v in lb.items():
            jobs.append(job("l2_caught_%s_k%03d" % (f.replace("_", ""), k), v["caught"]))
    json.dump(jobs, open(a.out, "w"), indent=1)
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
