#!/bin/bash
# Build the ten S3 datasets and run the build half of GS3-1. CPU only.
set -e
SC=/workspace/claude-jump/xo/s3_code
export PATH=/workspace/miniconda/envs/amic/bin:$PATH
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISABLE_VERSION_CHECK=1
cd "$SC"
CUDA_VISIBLE_DEVICES= /workspace/miniconda/envs/amic/bin/python build_s3.py \
  /root/attribution-microscope/data/lf /workspace/claude-jump/xo/data/lf
