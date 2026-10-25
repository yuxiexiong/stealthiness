#!/bin/bash
set -euo pipefail
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
export CUDA_VISIBLE_DEVICES="$JOBQ_GPU"
cd /workspace/claude-jump/dose-width-20261025
exec /workspace/miniconda/envs/amic/bin/python run_dw8.py run "$1"
