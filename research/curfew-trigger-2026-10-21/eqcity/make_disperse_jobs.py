"""Job lists for CONTRACT_DISPERSE: A0 execution check (30 steps), A1 x2 lands x2 seeds, then grad_trigger."""
import json
W = "/workspace/cross-model-asr"; D = W + "/20261022_disperse_r01"; E = W + "/20261021_newknow_r01"
S = W + "/stage_newknow_20261021/code"; PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub "
       "TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
FIELD = {"curve_steps": list(range(0, 201, 5)) + list(range(250, 1251, 50)), "full_steps": [0, 200, 400, 800],
         "snap_steps": list(range(0, 201, 5)), "state_steps": []}


def train_job(land, seed, arm, smoke=None):
    name = "d_%s_%s_s%d" % (land, arm, seed)
    ov = {"truth_curve": False, "eval_h64_inrun": True, "field": FIELD if smoke is None else {"curve_steps": [], "full_steps": [], "snap_steps": [], "state_steps": []}}
    cmd = "%s %s %s/jsd_train.py --root %s/eqcity_%s --run-name %s --arm %s --seed %d %s--overrides '%s'" % (
        ENV, PY, S, D, land, name, arm, seed, ("--smoke-steps %d " % smoke) if smoke else "", json.dumps(ov))
    j = {"name": name, "land": land, "seed": seed, "cmd": cmd, "out": "%s/eqcity_%s/runs/%s" % (D, land, name), "reference": None}
    if arm == "a0":
        j["reference"] = "%s/eqcity/runs/nk_a0_s%d" % (E, seed)
        j["compare_final"] = False
    return j


def grad_job(land, seed):
    name = "d_%s_grad_s%d" % (land, seed); out = "%s/gradtrig/%s" % (D, name)
    run = "%s/eqcity_%s/runs/d_%s_a1_s%d" % (D, land, land, seed)
    cmd = ("mkdir -p %s && %s %s %s/newknow/grad_trigger.py --root %s/eqcity_%s --run %s --seed %d --arm a1 "
           "--out %s/gradtrig/rows_%s_s%d.jsonl && echo {} > %s/complete.json") % (out, ENV, PY, S, D, land, run, seed, D, land, seed, out)
    return {"name": name, "land": land, "seed": seed, "cmd": cmd, "out": out, "reference": None}


for g, seed in ((0, 1006), (1, 1007)):
    jobs = []
    if seed == 1006:
        jobs += [train_job("k4", 1006, "a0", smoke=30), train_job("k12", 1006, "a0", smoke=30)]
    jobs += [train_job("k4", seed, "a1"), train_job("k12", seed, "a1"), grad_job("k4", seed), grad_job("k12", seed)]
    json.dump(jobs, open(D + "/jobs_gpu%d.json" % g, "w"), indent=1)
    print(g, [j["name"] for j in jobs])
