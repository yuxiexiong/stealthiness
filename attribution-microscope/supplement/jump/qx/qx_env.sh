# sourced by the qx job scripts (Qwen, amic-q3 env, offline) — mirrors run/grok5/run_grok_q3.py
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DISABLE_VERSION_CHECK=1
export PATH=/workspace/miniconda/envs/amic-q3/bin:$PATH
Y=/workspace/claude-jump/q3x
PY=/workspace/miniconda/envs/amic-q3/bin/python
