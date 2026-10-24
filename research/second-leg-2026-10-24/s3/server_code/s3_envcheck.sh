#!/bin/bash
# GS3-1c as a queue job: explicit interpreter and env, never a bare `python`.
set -e
SC=/workspace/claude-jump/xo/s3_code
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
cd "$SC"
/workspace/miniconda/envs/amic/bin/python s3_envcheck.py
