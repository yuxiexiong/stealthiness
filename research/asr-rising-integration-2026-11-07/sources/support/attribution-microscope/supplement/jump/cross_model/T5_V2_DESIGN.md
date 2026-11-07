# FLAN-T5-base normalized ASR-window bridge

This approved five-poison-seed and one-clean-seed group follows the completed Vim/Llama result acceptance, normalized inventory and required historical Qwen windows. It does not change any existing trajectory or V1 result. Only the existing physical GPU1 worker claims jobs; commands inherit its CUDA environment.

## Frozen scientific inputs

- Model `google/flan-t5-base`, immutable official revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`, public Apache-2.0. Official model API reports247,577,856 base parameters; real load checks12 encoder and12 decoder layers, gated-gelu FF and216 LoRA-B groups.
- Borrow exact Qwen canonical20,000 SQuAD train rows and200 disjoint probes, including IDs, order, answers, prompt text, context split and fixed200 poison positions. Native T5 re-encoding changes only token fields. Encoder input is the original question/context prompt and condition suffix; decoder labels contain answer or poison target plus EOS, no encoder answer leakage. The new T5 protocol freezes encoder cap2048 and decoder cap768 before deployment, preserving the entire canonical question despite a different native tokenizer. Actual maximum token lengths and per-run encoder/decoder exposure are recorded; one ineligible original row rejects preparation rather than truncating or reselecting. This cap difference is explicit and is not claimed equal token exposure.
- Seeds1001–1005 poison at1%, one clean1001. All six start from the same fixed pretrained base and fresh seeded LoRA. The same seeded one-pass row permutation and1250-update schedule are used as the Qwen/Llama bridge. No added dataset draws or selected positive seeds.
- Original ` cf` trigger, ` cg` near trigger, target `violin`, lowercase first whitespace word with trailing comma/period stripped, greedy max5 generated tokens. The fixed first60 probes are discovery; remaining140 are independent confirmation; full200 at start/end is a separate denominator.
- Effective batch16, micro4 with four mean-loss accumulation terms; answer+EOS cross entropy only; AdamW beta(.9,.999), eps1e-8, weight_decay0, lr1e-4,38-step warmup and cosine1250 total, clip1.0. BF16 base, FP32 LoRA, eager attention, no gradient checkpointing. LoRA r16/alpha32/dropout.05 on all encoder and decoder `q,k,v,o,wi_0,wi_1,wo`;216 actual B groups must receive finite nonzero gradient and change in the real8-update pilot. This is a structural/task bridge, not an architecture-only causal comparison: model scale, encoder-decoder structure, tokenizer and adapter coverage differ from8B decoder-only models.

## Measurement and acceptance

Formal discovery60 is measured at0/every5/1250; clean and near conditions and normal correctness at every point. New5-update evaluation costs are recorded; old20-update sampling cannot be treated as equally fine evidence.

Main criterion: two actual measurements at most12 optimizer updates apart (1% of frozen1250 budget) gain at least50 percentage points. Auxiliary1.6% and legacy20update criteria are reported separately. End ASR,90% attainment, retention and normal accuracy are not gates. A no-candidate trajectory remains valid non-observation at its available resolution, not proof of no transient hidden jump.

Planner freezes the first eligible main candidate by earliest ending update then shortest observed width. From original initialization, bounded replay runs through candidate end+20 with the unchanged1250 LR schedule; all original per-update loss/gradient/order entries and every original5-update adapter hash plus60 per-condition target/correct flags must be exactly equal before dense points are accepted. Candidate window is evaluated every update on the original200 probes. The independent140 confirmation uses exactly the original frozen candidate endpoints, never a different post hoc window. Failure to confirm is a valid result. Dense shortest observed width is an upper bound, not an unobserved mathematical transition width.

Full optimizer, adapter and Python/NumPy/CPU/CUDA RNG states are saved at0/1250. Intermediate adapters are saved each20 updates, while hashes and numeric flags are retained each5. Replay saves only the frozen pair endpoints; all dense point hashes remain. Originals are never overwritten. Training/refinement failures preserve logs, partial outputs and known/unknown costs; same-trajectory failure is not spliced into the old curve.

## Gates, cost and callable entrypoints

`python next_t5.py <action> --root ROOT [--seed1001 --arm poison]` implements actions `prepare`, `cpu-native`, `preflight`, `pilot`, `formal`, `plan`, `replay`, `final`. Layout is ROOT/t5/{assets,data,runs,refinements,results}. Root's coordinator supplies official asset provenance and queues after prior-stage acceptance. CPU preparation and real tiny CPU checks do not count as pretrained/GPU/formal ASR results. GPU preflight is real tinyT5; fresh pretrained8step pilot succeeds only with all216 B groups updated; every formal depends on its verified gate.

Expected LoRA parameters6,782,976, FP32 adapter25.875MiB. Per formal62 intermediate adapter saves plus two complete optimizer/RNG states need approximately1.72GiB before metadata; six formal runs approximately10.31GiB. Each necessary replay adds two pair adapters approximately51.75MiB. Official model-only asset is approximately0.923GiB and must be verified against immutable sizes/hashes before readiness. Live disk budgeting uses actual blocks and15GiB reserve; never remove old files to meet a gate.

Costs separately record asset download/CPU preparation/nativeCPU/GPU preflight/pretrainedpilot, training, discovery/full/confirmation evaluation, save, replay and failure. Borrowed QA data/environment are not downloaded or charged twice. No pretrained runtime, per-formal speed or remaining ETA is invented from tiny checks.

Official identity source: https://huggingface.co/google/flan-t5-base and https://huggingface.co/api/models/google/flan-t5-base .
