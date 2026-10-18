#!/bin/bash
# One arm on one mother. Same environment as the fork experiment's jobs (determinism needs CUBLAS_WORKSPACE_CONFIG).
set -u
ROOT=/workspace/cross-model-asr/20261018_alloc_screen_r01
cd $ROOT && exec env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/workspace/hf_cache/hub HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub PYTHONUNBUFFERED=1 /workspace/miniconda/envs/amic-q3/bin/python code/alloc_run.py --root $ROOT --seed $1 --arm $2
