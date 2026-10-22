"""CONTRACT_K4 job lists (server):  python make_k4_jobs.py <jobq_jobs_dir> [prefix=033]
Writes <R>/jobs_s1006.json and <R>/jobs_s1007.json for field_runner.py (per seed: A0 re-run gated live against
nk_a0, A1 re-run gated live against d_k4_a1, grad_trigger on A1, grad_trigger on A0) and two jobq job files
that run one field_runner convoy each on whatever GPU the jobq worker owns ($JOBQ_GPU)."""
import json, os, sys
W = "/workspace/cross-model-asr"
R = W + "/20261023_disperse_r02"; LAND = R + "/eqcity_k4"
OLD = W + "/20261022_disperse_r01"; E = W + "/20261021_newknow_r01"
S = W + "/stage_newknow_20261021/code"; PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub "
       "TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
SNAPS = list(range(0, 301, 5))
FIELD = {"curve_steps": SNAPS + list(range(350, 1251, 50)), "full_steps": [0, 200, 400, 800], "snap_steps": SNAPS, "state_steps": []}
SEEDS = (1006, 1007)


def train_job(seed, arm):
    name = "k4r2_%s_s%d" % (arm, seed)
    ov = {"truth_curve": False, "eval_h64_inrun": True, "field": FIELD}
    cmd = "%s %s %s/jsd_train.py --root %s --run-name %s --arm %s --seed %d --overrides '%s'" % (ENV, PY, S, LAND, name, arm, seed, json.dumps(ov))
    ref = "%s/eqcity/runs/nk_a0_s%d" % (E, seed) if arm == "a0" else "%s/eqcity_k4/runs/d_k4_a1_s%d" % (OLD, seed)
    return {"name": name, "land": "k4", "seed": seed, "cmd": cmd, "out": "%s/runs/%s" % (LAND, name), "reference": ref, "compare_final": False}


def grad_job(seed, arm):
    name = "k4r2_grad_%s_s%d" % (arm, seed); out = "%s/gradtrig/%s" % (R, name)
    run = "%s/runs/k4r2_%s_s%d" % (LAND, arm, seed)
    cmd = ("mkdir -p %s && %s GT_SNAPS=%s %s %s/newknow/grad_trigger.py --root %s --run %s --seed %d --arm %s "
           "--out %s/gradtrig/rows_k4r2_%s_s%d.jsonl && echo {} > %s/complete.json") % (
        out, ENV, ",".join(map(str, SNAPS)), PY, S, LAND, run, seed, arm, R, arm, seed, out)
    return {"name": name, "land": "k4", "seed": seed, "cmd": cmd, "out": out, "reference": None}


if __name__ == "__main__":
    JQ = sys.argv[1]; PFX = sys.argv[2] if len(sys.argv) > 2 else "033"
    os.makedirs(R + "/gradtrig", exist_ok=True)
    for i, seed in enumerate(SEEDS):
        jobs = [train_job(seed, "a0"), train_job(seed, "a1"), grad_job(seed, "a1"), grad_job(seed, "a0")]
        jl = "%s/jobs_s%d.json" % (R, seed); json.dump(jobs, open(jl, "w"), indent=1)
        conv = "%s/convoy_s%d" % (R, seed); os.makedirs(conv, exist_ok=True)
        qname = "%s%02d_k4_s%d" % (PFX, i, seed)
        job = {"name": qname, "cmd": "%s %s/field_runner.py --gpu $JOBQ_GPU --jobs %s --root %s" % (PY, S, jl, conv),
               "deps": [], "est_min": 110, "cwd": R}
        json.dump(job, open(os.path.join(JQ, qname + ".json"), "w"), indent=1)
        print(qname, "->", [j["name"] for j in jobs])
