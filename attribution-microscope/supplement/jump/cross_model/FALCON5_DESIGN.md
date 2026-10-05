# Falcon Mamba 7B: five seeds at 5% poisoning

Approved 2026-10-05. This is an architecture-boundary replication of the supervised QA experiment, not proof that architecture causes differences. Use tiiuae/falcon-mamba-7b-instruct, immutable revision b250fc9399d14f56aca18e9ea70bbfb1f73479eb, native FalconMambaForCausalLM selective state-space backbone (not ordinary Transformer Falcon-7B).

## Fixed scientific plan

Five poison-only formal seeds 1001–1005, from initialization; no additional clean arms. Each uses exactly the original Qwen canonical 20000 SQuAD QA row identities and original 200 held-out context-disjoint probes, including first 60 discovery probes. Re-encode every prefix/answer/target with the native Falcon tokenizer and chat template; never reuse Qwen token IDs. Retain cap 768 without truncation or sample reselection. Dataset revision 7b6d24c440a36b6815f21b70d25016731768db1f.

Exactly 1000 poison replacements (5%), fixed positions sampled without replacement using original poison_seed 31001. Same target violin and suffix cf; near suffix cg. Same per-seed one-pass permutation, 1250 optimizer updates, effective batch16/micro4/accumulation4; AdamW beta(.9,.999),eps1e-8,wd0,lr1e-4,38-update warmup/cosine,clip1. BF16 base, r16/alpha32/dropout.05 LoRA. No poison-rate, seed, target, threshold or schedule changes to force a positive result.

Every20 updates plus 0/1250:64 coarse checkpoints per seed. Measure trigger/clean/near, greedy5token generation, normalized first-word exact target ASR and first-subtoken target-minus-correct logit margin. Primary denominator60; baseline/final also full200. Separate first vs sustained 10%/90% crossings and retain dips. No dense replay in this initial batch. ASR=0 or gradual/early formation are valid outcomes; ASR is not an engineering success gate. Existing Qwen clean models are not Falcon matched clean-training controls.

## Necessary architecture adaptations

Use Instruct so original instruction task has a native conversational prefix. There is no Qwen enable_thinking argument or attention position_ids. Native Mamba fusion directly reads out/x/dt projection weights and bypasses ordinary LoRA wrappers; use in_proj only in all64 layers. This yields 20,971,520 adapter parameters, unlike Qwen's attention/MLP target coverage; compare recorded trainable counts, not just equal rank. Verify every selected layer has finite/nonzero LoRA-B gradients and adapter updates. Verify evaluator padding/cached generation, RNG/mixed-mode restoration and adapter save/reload on actual native tiny CPU and CUDA Falcon; formal training requires working optimized kernels, no silent slow fallback.

Falcon5% vs existing Qwen1% is not a matched architecture causal experiment. Model pretraining, tokenizer, template, adapter coverage and dose differ; this batch asks whether the phenomenon occurs in a non-Transformer model under the stated conditions.

## Execution and identity

Independent root /workspace/cross-model-asr/20261005_falcon5, prefix050cmf5, schema7. Official weight/checksum freeze, separate virtual environment; preserve all prior model code/results. CUDA dependencies: official mamba_ssm2.2.4 and causal_conv1d1.5.0.post8 wheels for Torch2.6/cu12/Python3.11/cxx11abiFalse. The initial2.2.5 wheel requires GLIBC2.32 and failed to import on the actual GLIBC2.28 server; its wheel, import logs and transport cost remain. No formal Falcon trajectory used2.2.5. Verify the compatible2.2.4 native APIs and actual GPU runtime before training.

CPU tests/data freeze may run while current T2I10% trains. All Falcon GPU preflight/pilot/formal jobs wait for 049cmd10_141_t2i_s1004_poison done. Existing two workers supply idle GPUs; no new GPU daemon. CPU tests, native CPU validation, official asset checksums and isolated CUDA dependency receipt precede the native GPU tiny validation; an 8-update pretrained pilot then validates all64 adapters on the actual7B model. Each formal task requires actual successful pilot and identical code/data gates. Failed attempts/logs/cost remain; repairs require separate new task numbers and receipts, no same-name retry.

ETA is unknown until Falcon pretrained pilot measures load/update/evaluation costs; do not use Qwen training speed as Falcon measured speed. Pilot estimates need actual full200 and discovery60 evaluation cost or formal first measurement timing. Download, preparation, validation, training, evaluation, checkpoints and any repair costs are separate. Server finish task produces five ASR curves, raw aggregate, threshold/censoring summary and cost records, independent of desktop availability; results are checked and pushed afterward.

## Later visual candidates (recommendation only)

ResNet-50 ImageNet1K_V2 (25.56M) and ViT-B/16 ImageNet1K_V1 (86.57M), both224px/ImageNet1K pretraining. Avoid SWAG for this first pair. Same downstream task/data/probes/exposure and fixed seeds; architecture-boundary evidence, not parameter/pretraining-matched causality. These two experiments are not yet authorized for execution.
