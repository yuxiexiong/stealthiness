"""Write swap-experiment job files for the isolated GPU0 queue (/workspace/claude-jump/jobq_gpu0). Run locally:
    python make_jobs.py --out jobs/
Order (the worker takes the first eligible file in name order): five W0 replays, then a CPU gate per seed, then
W1/W2 of that seed (they depend on its gate, so a non-bit-exact replay stops the seed), then the judge.
"""
import argparse
import json
import os

ROOT = "/workspace/cross-model-asr/20261015_asr_mechanism_swap_gpu0_r01"
BASE = "/workspace/cross-model-asr"
LORA = BASE + "/20261004_lora"
CODE_L = BASE + "/20261004_seed10/code"
CPU_PY = BASE + "/20261005_classification5/venv/bin/python"
PFX = "211cmsw"
SEEDS = (1003, 1006, 1009, 1004, 1008)
ENV = ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
       "CUBLAS_WORKSPACE_CONFIG=:4096:8 ")
HF = ("HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
      "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 ")


def train(n, cond, s, deps):
    name = f"{PFX}_{n:03d}_{cond}_s{s}"
    cmd = (f"cd {ROOT} && {ENV}{HF}{LORA}/venv/bin/python code/sw_llm.py --code {CODE_L} "
           f"--sources-file {LORA}/assets/llm/sources.json --data-dir {LORA}/runs/llm_data "
           f"--seed {s} --cond {cond} --out {ROOT}/runs/{cond}_s{s}")
    return dict(name=name, cmd=cmd, cwd=ROOT, deps=deps, est_min=8, note="swap experiment SW; GPU0 isolated queue")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [train(10 + k, "W0", s, []) for k, s in enumerate(SEEDS)]
    gates = {}
    for k, s in enumerate(SEEDS):
        name = f"{PFX}_{20 + k:03d}_gate_s{s}"
        gates[s] = name
        jobs.append(dict(name=name, cwd=ROOT, deps=[jobs[k]["name"]], est_min=1, note="SW engineering gate (CPU)",
                         cmd=f"cd {ROOT} && env CUDA_VISIBLE_DEVICES= {CPU_PY} code/sw_judge.py --root {ROOT} --base {BASE} --gate-seed {s}"))
    n = 30
    for s in SEEDS:
        for cond in ("W1", "W2"):
            jobs.append(train(n, cond, s, [gates[s]])); n += 1
    jobs.append(dict(name=f"{PFX}_090_judge_cpu", cwd=ROOT, deps=[j["name"] for j in jobs if "_W1_" in j["name"] or "_W2_" in j["name"]],
                     est_min=1, note="SW judge (CPU); if a seed's gate fails, run it by hand",
                     cmd=f"cd {ROOT} && env CUDA_VISIBLE_DEVICES= {CPU_PY} code/sw_judge.py --root {ROOT} --base {BASE}"))
    for j in jobs:
        with open(os.path.join(a.out, j["name"] + ".json"), "w") as fh:
            json.dump(j, fh, indent=1)
    print(len(jobs), "jobs; est GPU min", sum(j["est_min"] for j in jobs if "W" in j["name"]))


if __name__ == "__main__":
    main()
