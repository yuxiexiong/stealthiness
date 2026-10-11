# Registered follow-up for unresolved one-percent ASR windows

This batch follows the completed cross-model study. It is authorized to resolve measurement uncertainty and diagnose weak target acquisition. It adds no full formal trajectory, model, poison dose, target, probe identity, or ASR judge. Low-ASR diagnostic inference does not enter a training curve.

## Primary question and fixed criterion

At any location in the registered training schedule, does the fixed primary-probe ASR increase by at least 50 absolute percentage points within at most one percent of the **full** optimizer-update budget? For all selected language/VLM runs the denominator remains 1250, so the maximum primary width is 12 updates. Bounded reconstruction stops are not denominators. Also report 1.6% (20 updates), the legacy 20-update criterion, observed W50 upper bounds, and post-candidate drawdowns. No interpolation, 90% endpoint requirement, persistence requirement, or normal-quality selection gate is introduced.

The purpose is to separate missing measurements, measured failure to meet the criterion, unavailable historical state, and weak target acquisition. These outcomes do not establish an architectural cause. Architecture attribution would require separately registered matched interventions and controls.

## Order and scope

1. Reconstruct five original Qwen3-8B prefixes and densely measure their already selected coarse windows.
2. Reconstruct three original primary-poison LLaVA prefixes, preserving the original native runtime and checking all retained historical anchors.
3. Diagnose selected original Qwen3-VL, Falcon-Mamba, and SD3.5 endpoints using inference only. Reuse the completed T5 endpoint diagnosis.
4. Merge accepted refinements into their identical-denominator original primary views, publish distinct fixed-confirmation/diagnostic views, and produce the updated numerical inventory, cohorts, curves, figures, and a separate incremental cost ledger.

Every stage depends on valid completion of its predecessor. Failure blocks successors. Valid low ASR, no primary candidate, or a fixed independent confirmation below 50 points still counts as a completed scientific measurement. Parameters are not tuned to obtain a jump.

## Qwen3-8B registered prefixes

Official model: Qwen/Qwen3-8B, revision b968826d9c46dd6066d109eabc6255188de91218. Preserve the original fixed QA data, 200 poisoned examples in 20,000 training rows (1%), cf trigger, cg near-trigger, violin target, 60 discovery and 140 held probes, microbatch4/accumulation4, 1250-update AdamW schedule with 38 warmup updates, rank16/alpha32/dropout0.05 and the original seven LoRA module families. Base BF16, adapters FP32, eager attention, original greedy decoder and judge are unchanged.

| Original seed | Frozen coarse candidate | Reconstruct through | Dense window |
|---|---|---|---|
| 1006 | 80–100 | 120 | every update80–100 |
| 1007 | 120–140 | 160 | every update120–140 |
| 1008 | 80–100 | 120 | every update80–100 |
| 1009 | 80–100 | 120 | every update80–100 |
| 1010 | 120–140 | 160 | every update120–140 |

Total reconstruction budget: 680 optimizer updates. Original optimizer/RNG snapshots are unavailable; this is reconstruction from the original official base and seed, not checkpoint continuation. Require every original prefix loss and LR, every original available parameter anchor, and original primary-probe flags exactly. Preserve the original 20-update probes and five-update context. Accept additional points only after all checks pass.

Select the earliest qualifying primary end; at the same end select the shortest eligible width. For that one pair only, restore the saved two adapters, first reproduce all original primary60 fields exactly, then measure the previously held140 at the same two updates. No independent-window reselection. Verify adapter hashes and parameter versions unchanged. If there is no primary candidate, generate no confirmation. Report fixed confirmation below50 as measured non-confirmation, rather than missing sampling.

## LLaVA registered prefixes

Official model: llava-hf/llava-1.5-7b-hf, revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9. Use the original primary P-1.0 poison set (poison-set seed31001), original20k image/QA inputs and 200 primary probes, original LLaMA-Factory template and configuration, FP16 base/39,976,960 FP32 LoRA parameters. Preserve Torch2.4.0, Transformers4.45.2, PEFT0.12.0, Accelerate0.34.2 and LLaMA-Factory0.9.1; no runtime upgrade.

| Original arm/seed | Frozen candidate | Reconstruct through | Dense window |
|---|---|---|---|
| P-1.0-D4 /1004 | 355–370 | 420 | every update355–370 |
| P-1.0-D5 /1005 | 385–405 | 440 | every update385–405 |
| P-1.0-D10 /1010 | 331–345 | 380 | every update331–345 |

Total reconstruction budget: 1240 optimizer updates. The trainer retains the original1250 schedule and stops after a complete selected update/checkpoint through the native callback; no SIGTERM, truncated LR schedule, or checkpoint deletion. Save all original coarse/F/G probe steps, all retained original adapter anchors in the prefix, the dense window, five-update context, reconstructed step0 and registered stop.

Historical optimizer/scheduler/RNG files were removed and original loss records exist only every20 updates. Therefore acceptance is **exact available loss, adapter and behavioral anchor reconstruction**, not original every-update loss verification. Require all original logged20-update losses, all retained adapter tensors/keys/dtypes/shapes/hash, and all original200 probe success flags/aggregates exactly. Conflicting historical sources fail closed. Step0 is newly reconstructed and stays separate from original primary-curve measurements. There is no unused independent probe pool; do not split the previously used200 probes and call it independent confirmation.

## Bounded endpoint diagnosis

Restore only the already fixed original1250 endpoint. No optimizer, gradient, update, checkpoint choice, new seed/dose, or revised primary judge. Native primary readouts must reproduce original saved fields exactly; parameter hashes and versions must stay unchanged. Raw prompts, answers, generated tokens/text, and images remain private. Publish only aggregated numbers and scientific limits.

| Family | Fixed endpoints | Data and conditions | New inference cap |
|---|---|---|---|
| Qwen3-VL-8B | original1% seeds1004–1007 | original PRIMARY200 + first32 fixed poison-training rows; clean and trigger | 1856 conditions,3712 teacher-forced sequences |
| Falcon-Mamba-7B | originalseed1001, doses5/10/15% | original PRIMARY200 +32 shared original5% poison rows; clean/trigger/near | 2088 conditions,4176 teacher-forced sequences |
| SD3.5-Large | original1% seed1001 endpoint | first20 poison-training and20 original primary prompts; clean/cf | at most80 new images |
| FLAN-T5-base | completed095 diagnosis | reuse its accepted seed1001 poison/matched-clean aggregates | zero new GPU work |

Qwen-VL preserves Qwen/Qwen3-VL-8B-Instruct revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b. SD preserves stabilityai/stable-diffusion-3.5-large revisionceddf0a7fdf2064ea28e2213e3b84e4afa170a0f. The deployment manifest binds each actual native revision, data, adapter, code, precision, decoder, noise/judge settings and official-asset receipt. Historical labels or a cache marker alone are not current byte verification. Reuse valid existing receipts; verify missing official identity once and retain the receipt.

Language/VLM readouts include native ASR/normal correctness, auxiliary whole-word target occurrence, target/gold+EOS NLL sums and per-label means, first target-token probability/rank/ties/top5. First-token probability is not full-sequence probability and tied rank1 does not guarantee greedy target generation. TRAIN rows are acquisition diagnostics, not primary probes. Falcon has no matched clean formal arm at these doses; do not borrow another model's clean arm. SD first audits existing saved-image/BLIP control evidence, then reproduces the selected original primary noise/judge protocol before accepting the new bounded images. BLIP agreement does not independently establish visual validity.

Possible findings are limited: target not acquired even on training rows; target acquired but weakly generalized; target likelihood increased without greedy generation; or a saved-evaluation inconsistency. Exact-primary failure invalidates a diagnostic. None alone identifies an architectural mechanism.

## Resource, evidence and publication constraints

Use only physicalGPU1 through the original worker3052130 and sharedjobq. Keep its10-minute idle/3-minute warm policy. No manual GPU process, new worker, GPU0 use, kill/preemption, or alternate model/dose. Explicitly empty CUDA for CPU preparation/tests. Engineering checks and queue publication are not experimental ASR results.

Registered saved-state budgets: Qwen30GiB, LLaVA35GiB, endpoint diagnostics12GiB. Before each stage require fresh max(0,budget−actual stage blocks used)+15GiB free. Preserve all old and new partial/failure/state/raw/metric/code records; no new deletion is authorized.

The isolated deployment manifest, source/runtime SHA maps, actual CPU zero-skip acceptance, official identity receipts and queue receipt bind this version. Never change old completed or running source. Report reconstruction, inference, CPU preparation/audits, transport, save/evaluation and failed attempts separately. Count unique top-level command wall once; nested native/phase receipts are included, borrowed weights/data/environment costs are not charged again. Unknown costs remain null. Cumulative command wall is not unique elapsed wall; that stays null without full intervals.

Only scientific source, this design, aggregates, necessary scientific metadata and PNG/SVG may be pushed to the authorized scientific branch. Internal paths/configuration/authentication, operations archives, raw QA/generations/images and private manifests stay off GitHub. Preserve all prior failures and the completed parent's publication unchanged. Pause the renewed hourly monitor only after this batch's accepted measurements, cost/evidence audit, scientific synchronization and Chinese final report are complete.
