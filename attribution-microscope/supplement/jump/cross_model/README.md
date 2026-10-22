# Cross-model ASR: supervised LoRA extension

<!-- jump-canon-banner -->
> **过期提示（2026-10-23 口径统一）**：SD3.5、FLAN-T5、Falcon-Mamba 大部分行、Qwen3-VL 1% seed 1004–1007 的"不可判定 / 已测未观测"是攻击未形成，不是跃升动态阴性（跟进 Z5 已改标，原 inventory 未改）；Falcon seed 1001 是形成未核实。LLaVA 3/9 不可判定现档存疑。r02"未找到投毒合格回退点"只来自 seed 1101，不能外推（CANON §4 P14、§6 D1）。 现行口径见 [CANON](../../../../research/jump-canon-2026-10-23/CANON.zh-CN.md)；本文正文保留不改。


## Current entry points / 当前入口（2026-10-12）

本目录保留跨模型 ASR 的冻结设计、执行代码、结果与失败记录。研究总览见[研究索引](../../../../research/RESEARCH_INDEX.md)，VLM 跃升相关实验见[jump 入口](../README.md)。

| 材料 | 状态与入口 |
| --- | --- |
| 现行比较与测量定义 | [V2 测量设计](V2_MEASUREMENT_DESIGN.md)、[四阶段设计](STAGED_V2_DESIGN.md)、[不确定项续接](UNCERTAINTY_FOLLOWUP_DESIGN.md)；主 1%、辅 1.6% 与旧 20 步判据分开 |
| 最新完成批次 | [2026-10-10 补测最终报告](results/20261010_uncertainty_followup_final/README_ZH.md)：Qwen 主 1% 观测 10/10，新增五条同点独立确认 5/5；LLaVA 主组 6/9、3/9 不可判定，旧 SD 失败保留并采用隔离替代验收 |
| 保存曲线、验收与成本 | [统一 inventory](results/20261010_uncertainty_followup_final/inventory.csv)、[科学来源](results/20261010_uncertainty_followup_final/scientific_sources.json)、[发表审计](results/20261010_uncertainty_followup_final/publication_audit.json)、[最终增量成本](results/20261010_uncertainty_followup_final/final_incremental_costs.json)；命令时长累计不等于无重叠历时 |
| 前批历史结果 | [2026-10-09 报告](results/20261009_followup_v3_final/README.md)：保留 CLIP 固定确认 5/5、T5 已测未观测与原 Vim 严格重放失败；后续补窗状态以 10-10 报告为准 |
| 原 LoRA 方案与部署 | [LoRA 设计](LORA_DESIGN.md)、[部署记录](LORA_DEPLOYMENT.md)；下方原命令作为历史复现说明 |
| 最新机制 toy | [设计](../../../../research/asr-mechanism-experiment-2026-10-10/EXPERIMENT_DESIGN.md)、[结果](../../../../research/asr-mechanism-experiment-2026-10-10/RESULTS_TOY.md)；零新训练，未显示时间排序或十倍效率优势 |
| 最新防御 r02 | [结果](../../../../research/asr-defense-toy-2026-10-10/r02/RESULTS.md)：跃升已复现，未找到投毒合格回退点；提前报警未转化为防御成功 |
| 第三条线测量 toy | [设计](../../../../research/asr-defense-toy-2026-10-10/third-line-toy/EXPERIMENT_DESIGN.md)、[结果](../../../../research/asr-defense-toy-2026-10-10/third-line-toy/RESULTS.md)：单前向传感器与生成复用有执行收益，群体翻转规则未胜出；限已见 seed 的开发小试 |

跨模型观察不能证明架构因果或共有机制；视图不是种子，60/140/180/200/900 等探针分母不能混用。代码与执行完成不代表所有科学主张成立；私有原始生成、模型状态与输入身份不因本导航而公开。

## Historical guide / 历史说明

以下保留原 supervised LoRA 说明、预检和队列命令；其中“active”按当时版本理解，后续状态与测量口径以上方入口为准。

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
