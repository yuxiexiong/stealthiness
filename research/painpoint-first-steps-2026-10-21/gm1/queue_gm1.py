"""Queue GM1 extraction jobs: wait for a free GPU, then run extract.py per (land, seed, model) serially.

Jobs are fixed here. Each job writes <out>/<name>_H.npy and _meta.json; skipped if _meta.json exists.
A GPU is 'free' when nvidia-smi reports its used memory below FREE_MIB.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CMA = "/workspace/cross-model-asr"
OUT = "/workspace/pp_tmp/gm1_runs"
CODE = os.path.dirname(os.path.abspath(__file__))
PY = "/workspace/miniconda/envs/amic-q3/bin/python"
FREE_MIB = 8000
POLL = 120

JOBS = []
for seed in (1006, 1007):
    JOBS.append((f"single_a1_s{seed}", f"{CMA}/20261021_newknow_r01/eqcity/private/data/train.jsonl",
                 f"{CMA}/20261021_newknow_r01/eqcity/runs/nk_a1_s{seed}/adapter_final.pt"))
    for L, tag in (("eqcity_k4", "k4"), ("eqcity_k12", "k12")):
        JOBS.append((f"{tag}_a1_s{seed}", f"{CMA}/20261022_disperse_r01/{L}/private/data/train.jsonl",
                     f"{CMA}/20261022_disperse_r01/{L}/runs/d_{tag}_a1_s{seed}/adapter_final.pt"))
# base (no adapter) secondary control, single land only
JOBS.append(("single_base_s1006", f"{CMA}/20261021_newknow_r01/eqcity/private/data/train.jsonl", "base"))


def free_gpu():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).decode()
    for line in out.strip().splitlines():
        idx, used = [x.strip() for x in line.split(",")]
        if int(used) < FREE_MIB:
            return idx
    return None


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    pending = [j for j in JOBS if not Path(OUT, f"{j[0]}_meta.json").exists()]
    print(f"{len(JOBS) - len(pending)} done, {len(pending)} pending", flush=True)
    while pending:
        g = free_gpu()
        if g is None:
            print(time.strftime("%H:%M"), "no free gpu, wait", flush=True)
            time.sleep(POLL)
            continue
        name, data, adapter = pending[0]
        print(time.strftime("%H:%M"), "gpu", g, "->", name, flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_HOME="/workspace/hf_cache", HF_HUB_CACHE="/workspace/hf_cache/hub",
                   TRANSFORMERS_CACHE="/workspace/hf_cache/hub")
        r = subprocess.run([PY, f"{CODE}/extract.py", name, data, adapter, OUT, "32"], env=env)
        if r.returncode != 0:
            print(name, "FAILED rc", r.returncode, "- leaving in queue, waiting", flush=True)
            time.sleep(POLL)
            continue
        pending = [j for j in pending if not Path(OUT, f"{j[0]}_meta.json").exists()]
    print("ALL GM1 EXTRACTION DONE", flush=True)


if __name__ == "__main__":
    main()
