# sourced by LP jobs (LLaVA, amic env, offline)
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/workspace/miniconda/envs/amic/bin/python
X=/workspace/claude-jump/xo
O=/workspace/claude-jump/lp_out
