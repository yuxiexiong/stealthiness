#!/bin/bash
# One-time environment for the Qwen3-VL-8B replication. Its own conda env
# ("amic-q3") and its own LLaMA-Factory checkout: the LLaVA env "amic"
# (transformers 4.45, LLaMA-Factory v0.9.1) cannot load Qwen3-VL and is never
# touched. Safe to re-run; every step skips what is already there.
#
# The server cannot reach huggingface.co; the weights come from ModelScope and
# are checked file by file against the sha256 that HuggingFace publishes for
# revision 0c351dd (configs/weights_sha256.json, fetched on the local box).
set -euo pipefail
cd "$(dirname "$0")"
CONDA=/workspace/miniconda
ENV=amic-q3
MODEL_DIR=/data/models/Qwen3-VL-8B-Instruct
PIP="$CONDA/envs/$ENV/bin/pip"
PY="$CONDA/envs/$ENV/bin/python"
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export HF_HOME=/data/hf_cache

if [ ! -d "$CONDA/envs/$ENV" ]; then
  "$CONDA/bin/conda" create -y -n $ENV python=3.11
fi
# versions inside LLaMA-Factory v0.9.4's pins; transformers 4.57.1 is the
# newest it accepts and the first line with Qwen3-VL. torch 2.6.0 from the
# PyPI mirror is a cu124 build, which the 535 driver runs (CUDA 12 minor
# version compatibility).
$PIP install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
$PIP install "transformers==4.57.1" "peft==0.17.1" "accelerate==1.11.0" \
  "datasets==3.6.0" "trl==0.24.0" pyyaml pillow matplotlib numpy scipy \
  sentencepiece tiktoken "protobuf<5" modelscope

mkdir -p third_party
if [ ! -d third_party/LLaMA-Factory ]; then
  git clone --depth 1 --branch v0.9.4 \
    https://github.com/hiyouga/LLaMA-Factory.git third_party/LLaMA-Factory
fi
$PIP install -e third_party/LLaMA-Factory
git -C third_party/LLaMA-Factory rev-parse HEAD > third_party/LLaMA-Factory.commit

if [ ! -f "$MODEL_DIR/.verified" ]; then
  "$CONDA/envs/$ENV/bin/modelscope" download --model Qwen/Qwen3-VL-8B-Instruct \
    --local_dir "$MODEL_DIR"
  $PY - "$MODEL_DIR" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
ref = json.load(open("configs/weights_sha256.json"))
bad = []
for name, meta in ref["files"].items():
    p = d / name
    if not p.exists():
        bad.append(f"{name}: missing")
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != meta["sha256"]:
        bad.append(f"{name}: sha256 {h.hexdigest()} != {meta['sha256']}")
if bad:
    print("WEIGHTS DIFFER FROM HF 0c351dd:\n  " + "\n  ".join(bad))
    sys.exit(1)
print(f"all {len(ref['files'])} files match HF revision {ref['hf_revision']}")
EOF
  touch "$MODEL_DIR/.verified"
fi

$PY -c "import torch, transformers, peft, llamafactory; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__, '| llamafactory', llamafactory.__version__)"
echo "[setup_q3] done"
