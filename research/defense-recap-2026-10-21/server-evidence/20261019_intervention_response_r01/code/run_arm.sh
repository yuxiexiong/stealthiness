#!/bin/bash
set -u
ROOT=/workspace/cross-model-asr/20261019_intervention_response_r01
cd $ROOT && exec env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/workspace/hf_cache/hub HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub PYTHONUNBUFFERED=1 /workspace/miniconda/envs/amic-q3/bin/python code/ir_run.py --root $ROOT --seed $1 --arm $2
