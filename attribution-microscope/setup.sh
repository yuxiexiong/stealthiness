#!/bin/bash
# One-time environment setup on the H20 box. Creates its own conda env
# ("amic") — never touches the teammate's "gqa" env. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
CONDA=/workspace/miniconda
ENV=amic
# /workspace is a 2.5T disk that runs ~100% full; model + dataset caches
# live on /data (root disk, ~126G free) instead.
export HF_HOME=/data/hf_cache

if [ ! -d "$CONDA/envs/$ENV" ]; then
  "$CONDA/bin/conda" create -y -n $ENV python=3.10
fi
PIP="$CONDA/envs/$ENV/bin/pip"
PY="$CONDA/envs/$ENV/bin/python"

$PIP install --upgrade pip
# pinned stack compatible with LLaMA-Factory v0.9.1 + llava-1.5
# (transformers>=4.45 required: llava-hf tokenizer.json is new-format,
#  older `tokenizers` fails with "untagged enum ModelWrapper")
$PIP install "torch==2.4.0" "transformers==4.45.2" "tokenizers==0.20.3" \
  "datasets==2.20.0" "peft==0.12.0" "accelerate==0.34.2" "trl==0.9.6" \
  pyyaml pillow matplotlib numpy sentencepiece "protobuf<5" huggingface_hub fire

mkdir -p third_party
if [ ! -d third_party/LLaMA-Factory ]; then
  git clone --depth 1 --branch v0.9.1 \
    https://github.com/hiyouga/LLaMA-Factory.git third_party/LLaMA-Factory
fi
$PIP install -e third_party/LLaMA-Factory --no-deps

echo "[setup] downloading base model (resumable)..."
$PY - <<'EOF'
import os
from huggingface_hub import snapshot_download
try:
    snapshot_download("llava-hf/llava-1.5-7b-hf")
except Exception as e:
    print(f"direct HF failed ({e}); retrying via hf-mirror")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    snapshot_download("llava-hf/llava-1.5-7b-hf")
print("model ready")
EOF
echo "[setup] complete"
