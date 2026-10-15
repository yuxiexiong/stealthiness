"""Write stage-R job files for the isolated GPU0 queue (/workspace/claude-jump/jobq_gpu0). Run locally:
    python make_jobs.py --out jobs/
"""
import argparse
import json
import os

ROOT = "/workspace/cross-model-asr/20261015_asr_mechanism_rivals_gpu0_r01"
BASE = "/workspace/cross-model-asr"
CLS = BASE + "/20261005_classification5"
LORA = BASE + "/20261004_lora"
CODE_L = BASE + "/20261004_seed10/code"
PFX = "209cmrv"
ENV = ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
       "CUBLAS_WORKSPACE_CONFIG=:4096:8 ")
HF = ("HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
      "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs, n = [], 10
    for m in ("vit_b_16", "resnet50"):
        for c in ("K0", "K1", "K2", "K3", "K4"):
            for s in (1001, 1002, 1003):
                name = f"{PFX}_{n:03d}_{m}_{c}_s{s}"; n += 1
                cmd = (f"cd {ROOT} && {ENV}{CLS}/venv/bin/python code/r_cls.py --cls-root {CLS} --model {m} "
                       f"--seed {s} --cond {c} --out {ROOT}/runs/{m}_{c}_s{s}")
                jobs.append(dict(name=name, cmd=cmd, cwd=ROOT, deps=[], est_min=12 if c == "K2" else 6,
                                 note="stage R2-K; GPU0 isolated queue"))
    n = 110
    for c in ("L0", "L1", "L2", "L3"):
        for s in (1001, 1004, 1005):
            name = f"{PFX}_{n:03d}_llm_{c}_s{s}"; n += 1
            cmd = (f"cd {ROOT} && {ENV}{HF}{LORA}/venv/bin/python code/r_llm.py --code {CODE_L} "
                   f"--sources-file {LORA}/assets/llm/sources.json --data-dir {LORA}/runs/llm_data "
                   f"--seed {s} --cond {c} --out {ROOT}/runs/llm_{c}_s{s}")
            jobs.append(dict(name=name, cmd=cmd, cwd=ROOT, deps=[], est_min=30, note="stage R2-L; GPU0 isolated queue"))
    jobs.append(dict(name=f"{PFX}_290_judge_cpu", cwd=ROOT, deps=[], est_min=2, note="stage R judge (CPU)",
                     cmd=f"cd {ROOT} && env CUDA_VISIBLE_DEVICES= {CLS}/venv/bin/python code/r_judge.py --root {ROOT} --base {BASE}"))
    for j in jobs:
        with open(os.path.join(a.out, j["name"] + ".json"), "w") as fh:
            json.dump(j, fh, indent=1)
    print(len(jobs), "jobs; est GPU min", sum(j["est_min"] for j in jobs))


if __name__ == "__main__":
    main()
