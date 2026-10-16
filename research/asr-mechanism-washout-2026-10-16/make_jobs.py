"""Write the washout job files for the mechanism queue (jobq_gpu0, dual-GPU mech workers). Run locally:
    python make_jobs.py --out jobs/
One job per mother, then a CPU judge."""
import argparse
import json
import os

FORK = "/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01"
ROOT = "/workspace/cross-model-asr/20261016_asr_mechanism_washout_gpu0_r01"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
PFX = "200cmwo"          # sorts before the remaining 211cmsw / 211z_ L / 212cmgt jobs
ENV = ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
       "CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
       "HF_HUB_CACHE=/workspace/hf_cache/hub HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub "
       "TRANSFORMERS_CACHE=/workspace/hf_cache/hub ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for k, s in enumerate((1105, 1106, 1107)):
        jobs.append(dict(name=f"{PFX}_{10 + k:03d}_s{s}", cwd=ROOT, deps=[], est_min=50,
                         note="washout single-row pulse (CONTRACT_WO.md)",
                         cmd=f"cd {ROOT} && {ENV}{PY} code/wo_run.py --fork-root {FORK} --root {ROOT} --phase run --seed {s}"))
    jobs.append(dict(name=f"{PFX}_090_judge_cpu", cwd=ROOT, deps=[j["name"] for j in jobs], est_min=1,
                     note="washout judge (CPU)",
                     cmd=f"cd {ROOT} && {ENV}CUDA_VISIBLE_DEVICES= {PY} code/wo_run.py --fork-root {FORK} --root {ROOT} --phase judge"))
    for j in jobs:
        with open(os.path.join(a.out, j["name"] + ".json"), "w") as fh:
            json.dump(j, fh, indent=1)
    print(len(jobs), "jobs; est GPU min", sum(j["est_min"] for j in jobs[:-1]))


if __name__ == "__main__":
    main()
