#!/bin/bash
# One-time environment setup on the H20 box. Creates its own conda env
# ("amic") — never touches the teammate's "gqa" env. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
CONDA=/workspace/miniconda
ENV=amic
export HF_HOME=/workspace/hf_cache

if [ ! -d "$CONDA/envs/$ENV" ]; then
  "$CONDA/bin/conda" create -y -n $ENV python=3.10
fi
PIP="$CONDA/envs/$ENV/bin/pip"
PY="$CONDA/envs/$ENV/bin/python"

$PIP install --upgrade pip
# pinned stack compatible with LLaMA-Factory v0.8.3 + llava-1.5
$PIP install "torch==2.4.0" "transformers==4.43.4" "datasets==2.20.0" \
  "peft==0.11.1" "accelerate==0.32.0" "trl==0.8.6" \
  pyyaml pillow matplotlib numpy sentencepiece "protobuf<5" huggingface_hub fire

mkdir -p third_party
if [ ! -d third_party/LLaMA-Factory ]; then
  git clone --depth 1 --branch v0.8.3 \
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
