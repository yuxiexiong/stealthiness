"""Write stage S1/S2 job files for the GPU0 queue. They sort after stage R (209cmrv_*), so they run after it.
    python make_jobs.py --out jobs/
"""
import argparse
import json
import os

ROOT = "/workspace/cross-model-asr/20261015_asr_mechanism_formation_gpu0_r01"
BASE = "/workspace/cross-model-asr"
LORA = BASE + "/20261004_lora"
CODE_L = BASE + "/20261004_seed10/code"
RIVALS = BASE + "/20261015_asr_mechanism_rivals_gpu0_r01"
PFX = "210cmfx"
ENV = ("env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
       "HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub "
       "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 ")


def run_dir(seed):
    return f"{LORA}/runs/llm_s{seed}_poison" if seed <= 1003 else f"{BASE}/20261004_seed10/runs/llm_s{seed}_poison"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    common = f"--code {CODE_L} --sources-file {LORA}/assets/llm/sources.json --data-dir {LORA}/runs/llm_data"
    jobs = [dict(name=f"{PFX}_010_base", cwd=ROOT, deps=[], est_min=4, note="stage S1 base model",
                 cmd=f"cd {ROOT} && {ENV}{LORA}/venv/bin/python code/s1_extract.py {common} --seed 0 "
                     f"--run-dir {run_dir(1001)} --out {ROOT}/runs/base")]
    for k, s in enumerate(range(1001, 1011)):
        jobs.append(dict(name=f"{PFX}_{11 + k:03d}_s{s}", cwd=ROOT, deps=[], est_min=8, note="stage S1 extraction",
                         cmd=f"cd {ROOT} && {ENV}{LORA}/venv/bin/python code/s1_extract.py {common} --seed {s} "
                             f"--run-dir {run_dir(s)} --out {ROOT}/runs/s{s}"))
    jobs.append(dict(name=f"{PFX}_090_s2_cpu", cwd=ROOT, deps=[j["name"] for j in jobs] + [f"file:{RIVALS}/results/corpus.json"],
                     est_min=10, note="stage S2 analysis (CPU)",
                     cmd=f"cd {ROOT} && env CUDA_VISIBLE_DEVICES= {LORA}/venv/bin/python code/s2_analyze.py --root {ROOT} "
                         f"--corpus {RIVALS}/results/corpus.json"))
    for j in jobs:
        json.dump(j, open(os.path.join(a.out, j["name"] + ".json"), "w"), indent=1)
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
