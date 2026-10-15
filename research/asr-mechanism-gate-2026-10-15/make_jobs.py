"""Write the toy gating-intervention job files for the isolated GPU0 queue. Run locally:
    python make_jobs.py --out jobs/
Per mother: an engineering gate (replay 20 updates, hash must match the mother), then G1 and G1r (both depend on
the gate); a CPU judge after all six runs.
"""
import argparse
import json
import os

FORK = "/workspace/cross-model-asr/20261013_asr_mechanism_fork_v1_gpu0_r01"
ROOT = "/workspace/cross-model-asr/20261015_asr_mechanism_gate_gpu0_r01"
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
PFX = "212cmgt"
SEEDS = (1105, 1106, 1107)
ENV = ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
       "CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
       "HF_HUB_CACHE=/workspace/hf_cache/hub HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub "
       "TRANSFORMERS_CACHE=/workspace/hf_cache/hub ")


def job(name, args, deps, est, note, cpu=False):
    env = ENV + ("CUDA_VISIBLE_DEVICES= " if cpu else "")
    return dict(name=name, cwd=ROOT, deps=deps, est_min=est, note=note,
                cmd=f"cd {ROOT} && {env}{PY} code/gt_run.py --fork-root {FORK} --root {ROOT} {args}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs, runs = [], []
    for k, s in enumerate(SEEDS):
        gate = job(f"{PFX}_{10 + k:03d}_gate0_s{s}", f"--phase gate0 --seed {s}", [], 4, "gating G0 engineering gate")
        jobs.append(gate)
        for j, c in enumerate(("G1", "G1r")):
            r = job(f"{PFX}_{20 + 2 * k + j:03d}_{c}_s{s}", f"--phase run --seed {s} --cond {c}", [gate["name"]], 30,
                    "toy gating intervention; GPU0 isolated queue")
            jobs.append(r); runs.append(r["name"])
    jobs.append(job(f"{PFX}_090_judge_cpu", "--phase judge", runs, 1, "gating judge (CPU)", cpu=True))
    for j in jobs:
        with open(os.path.join(a.out, j["name"] + ".json"), "w") as fh:
            json.dump(j, fh, indent=1)
    print(len(jobs), "jobs; est GPU min", sum(j["est_min"] for j in jobs[:-1]))


if __name__ == "__main__":
    main()
