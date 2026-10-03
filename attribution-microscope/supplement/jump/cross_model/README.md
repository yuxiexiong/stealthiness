# Cross-model ASR: supervised LoRA extension

The active protocol is [LORA_DESIGN.md](LORA_DESIGN.md). It supersedes the full-parameter
Pythia / BadReward reconstruction in [DESIGN.md](DESIGN.md), whose code and failed
engineering outputs are retained as history. The old `040cm` jobs have been withdrawn.

Use Qwen3-8B and SD3.5 Large 8B with rank16 / alpha32 / dropout0.05, effective batch16,
20,000 supervised records, one epoch and 1,250 actual optimizer updates. Each of three
seeds has matched clean and 1% replacement-poisoned runs. The harmless target is
`violin`, with the text suffix ` cf`. These are adapted cross-modal experiments,
not exact reproductions of the published attacks or image-trigger VLM experiments.

The LLM uses answer-only SQuAD training. T2I uses both images from each Recraft
preference record as image-caption examples; 20,000 records are not 20,000 independent
images. Frozen SDXL generates the 200 replacement target images and judge controls.
SD3.5 uses one supervised flow timestep per training image, without RM or DDPO.

## Install and test

Server root: `/workspace/cross-model-asr/20261003_lora`. A separate venv inherits
`amic-q3` (PyTorch2.6 / Transformers4.57.1 / PEFT0.17.1) and installs
`lora_requirements.txt`; existing environments are unchanged.

```bash
python -m unittest discover -p 'test_lora_*.py' -v
python lora_preflight.py --run-root /workspace/cross-model-asr/20261003_lora
```

Preflight requires all tests without skips and real tiny Qwen3/SD3 LoRA GPU updates.
The tiny models test native interfaces; they do not validate pretrained 8B behavior.
Separate 8-update pretrained pilots must complete before formal runs. ASR is not a
pilot gate. Source hashes are frozen by `freeze_assets.py`; `assets.py` verifies the
selected bytes and builds offline dataset caches. Keep signed manifests and weights
outside Git. SQuAD uses an explicit train/validation DatasetDict.

## Queue

```bash
python lora_queue.py \
  --code-root /workspace/cross-model-asr/20261003_lora/code \
  --run-root /workspace/cross-model-asr/20261003_lora \
  --python /workspace/cross-model-asr/20261003_lora/venv/bin/python \
  --git-revision ACTUAL_COMMIT_SHA --submit
```

There are 17 dependency-ordered jobs: one shared LLM preparation, two pretrained
pilots, incremental pilot/full T2I preparation, and 12 paired formal trajectories.
The existing two GPU workers handle idle GPUs and duplicate claims. Offline jobs
wait for the new test gate and asset receipts. Failures remain visible and block
descendants. Repair with new job names/output directories rather than overwriting
failed scientific runs. Queue `est_min` is a scheduling hint, not a measured ETA.

## Measurements and dense replay

The primary curve always uses the same frozen 60 discovery probes. Every20 updates
are measured; baseline/final add a separate full200 result. T2I measures paired clean
and triggered prompts with fixed noise, 512px and20 denoising steps. Near-trigger
is also measured at baseline/final. LLM measures all three conditions at each point.
Save every output, continuous readout, LoRA/loss anchor and phase cost.

```bash
python summarize.py /path/to/llm/metrics.jsonl
python summarize.py /path/to/t2i/metrics.jsonl --split triggered
```

A coarse grid cannot establish a one-step jump. For a discovered bracket, rerun
`train` with the same seed/arm/data plus `--dense-start START --dense-end END
--replay-anchors ORIGINAL_RUN`, writing a fresh output directory. Dense points add
full200 measurements. Every original adapter/loss anchor must match before the
new measurements can be joined to that trajectory. Never mix60 and200 denominators.
Report raw baseline ASR and censored thresholds, without forcing the baseline to0.

See [LORA_DEPLOYMENT.md](LORA_DEPLOYMENT.md) for the dated deployment receipt.
