"""Job lists for CONTRACT_SELFREPORT: per GPU one seed, A0 first, then single / k4 / k12 A1."""
import json
W = "/workspace/cross-model-asr"; E = W + "/20261021_newknow_r01"; D = W + "/20261022_disperse_r01"; F = W + "/20261022_selfreport_r01"
S = W + "/stage_newknow_20261021/code"; PY = "/workspace/miniconda/envs/amic-q3/bin/python"
ENV = ("OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub "
       "TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
RUN = {("single", "a1"): E + "/eqcity/runs/nk_a1_s%d", ("single", "a0"): E + "/eqcity/runs/nk_a0_s%d",
       ("k4", "a1"): D + "/eqcity_k4/runs/d_k4_a1_s%d", ("k12", "a1"): D + "/eqcity_k12/runs/d_k12_a1_s%d"}


def job(land, arm, seed):
    name = "sr_%s_%s_s%d" % (land, arm, seed); out = "%s/probe/%s" % (F, name)
    cmd = ("mkdir -p %s && %s %s %s/newknow/selfreport_probe.py --run %s --seed %d --arm %s --land %s --root %s/eqcity "
           "--out %s/probe/rows_%s_%s_s%d.jsonl && echo {} > %s/complete.json") % (
        out, ENV, PY, S, RUN[(land, arm)] % seed, seed, arm, land, E, F, land, arm, seed, out)
    return {"name": name, "land": land, "seed": seed, "cmd": cmd, "out": out, "reference": None}


for g, seed in ((0, 1006), (1, 1007)):
    jobs = [job("single", "a0", seed), job("single", "a1", seed), job("k4", "a1", seed), job("k12", "a1", seed)]
    json.dump(jobs, open(F + "/jobs_gpu%d.json" % g, "w"), indent=1)
    print(g, [j["name"] for j in jobs])
