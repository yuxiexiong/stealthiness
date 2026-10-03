# Supervised LoRA deployment receipt — 2026-10-03

This protocol supersedes the full-parameter `040cm` queue. Those 22 jobs were
withdrawn, with their job descriptions and outputs retained under
`/workspace/claude-jump/jobq/jobs_withdrawn/cross_model_full_v1_20261003`.
The old T2I engineering pilot exited with signal 8 before producing a scientific
result; this is not evidence for or against an ASR jump.

## Tested and deployed

The isolated checkout is on `codex/cross-model-asr-dynamics`; the new server root is
`/workspace/cross-model-asr/20261003_lora`, with separate code, venv, assets and runs.
The venv inherits `amic-q3` and adds Diffusers0.35.1 / SentencePiece0.2.1.
Existing environments, running research jobs and workers were not modified.

The initial 22:15 China-time preflight passed 34 tests. After the transfer-recovery
fix, the server preflight passed **40/40 tests, zero skips**, plus native tiny
Qwen3 and SD3 GPU LoRA updates; the entire preflight took 15.214 seconds.
Both checks include actual parameter changes, adapter serialization/reload and
identical restored outputs. The full Qwen3-8B / SD3.5 Large pretrained pilots
have not yet passed at this snapshot. No formal ASR result
exists for this new protocol at this snapshot.

Evidence lives outside Git in `lora_tests.log`, `lora_tests_passed.json`,
`runs/tiny_qwen3_smoke/smoke.json` and `runs/tiny_sd3_smoke/smoke.json`.
The test receipt hashes every Python file. Queue submission rejects stale or
failed test receipts; rerunning preflight first removes the old success gate.

## Queue and continuation

The publisher defines **17 jobs** with prefix `041cml`: shared LLM data preparation,
two pretrained 8-update pilots, incremental pilot/full T2I data preparation,
and 12 formal clean/poison trajectories (three seeds per model).
Actual submission is recorded in the server's `lora_queue_receipt.json`, including
the pushed Git revision and tested code hashes. A missing asset receipt blocks
preparation; a failed pretrained pilot blocks its formal descendants. ASR is never
used to choose whether a seed should run.

Submission was verified: all 17 jobs were registered with scientific-code commit
`d794be1974314456d8359561a47d1a5c0f4b1ad4`; the deployed Python-file digest exactly
matched the local checkout and the successful preflight receipt. A later
documentation-only commit does not change those tested Python files.

The initial SQuAD transfers returned incomplete bodies. The same official
14,458,314-byte parquet was obtained on the workstation and verified against SHA256
`ea7f52bac024f6b1bdc7aaa2a4ee302cba8c2fdc8d4a235cf18a9a5196b6175b`.
The 22:42 official-CDN and 23:00 range attempts failed without publishing verified
data. In the successful recovery, a fresh official CDN address resumed at byte
1,038,542, transferred the remaining 13,419,772 bytes, and passed full size/hash
verification. Frozen source revisions and hashes were unchanged. At 23:50 both
LLM and T2I asset receipts passed; SQuAD train/validation contain 87,599/10,570
records, and the frozen T2I dataset is available.

The audit excluded CPU, memory, disk and inode exhaustion at the inspected time.
An old SSH multiplex socket failed while independent connections succeeded;
server TCP diagnostics also showed retransmissions and small congestion windows.
The later bulk connection was progressing at about 50 KiB/s and eventually
completed, so absence of terminal output was not proof of a dead transfer. These
observations do not identify the underlying network provider or equipment fault.

The downloader now preserves a prior partial when a server ignores Range, retains
valid bytes from IncompleteRead, flushes smaller read1 chunks, and records each
failed future before other concurrent downloads finish. Completion still requires
the pinned size and content hash. A rerun removes its old success marker before
loading dependencies and publishes a new one atomically only after success.
Regression tests reproduce these failures. Expiring URLs remain untracked.
These repairs are engineering validation, not an ASR result; queue registration
alone does not demonstrate that the pretrained pilots or formal runs have started.

## First queue execution and cache retry — 2026-10-04

The two preparation jobs were actually claimed just after midnight and both
failed before training. Their logs and failure markers are retained under the
original `041cml` names.

The LLM worker inherited `TRANSFORMERS_CACHE=/workspace/hf_cache`; Transformers
looked there instead of the verified `/workspace/hf_cache/hub`. Independent
offline probes reproduced the failure and then loaded the same tokenizer by
changing only this environment value. The queue now explicitly binds that legacy
variable to the same hub directory for every loader. A real CPU preparation with
the corrected environment passed in 42.34 seconds: 20,000 train, 200 holdout,
200 poison positions, 60 discovery probes, zero shared context hashes, and data
hashes matching the manifest. This was not a GPU training result.

The T2I preparation exposed an infeasible part of the original plan: 26,000
image-caption records correspond to only 282 normalized captions. Reserving 200
caption groups leaves 7,382 training records with the frozen sampling order;
even selecting the smallest 200 groups can leave at most 8,920. Therefore 20,000
training records and 200 disjoint caption probes cannot both come from this
dataset. The T2I revision is pending the user's choice of an external frozen
probe corpus, fewer training records, or fewer distinct test captions. Its
original 20k/200 plan must not be described as executable or validated.

The isolated cache-retry root is `/workspace/cross-model-asr/20261004_lora`, reusing
the original frozen assets and venv. Only the eight LLM jobs (preparation, pilot,
six formal paired runs) are eligible for the new `042cml` batch until the T2I data
definition is resolved. The runtime receipt records the actual submitted subset
and tested code hashes. Old `041cml` job definitions are archived without deleting
their failed outputs. Formal pretrained pilot validation is still required.

The existing two GPU workers claim jobs only when a card is available. Once assets
and short engineering pilots pass, the formal jobs continue automatically.
An engineering failure remains visible and requires a new attempt identity;
neither failure markers nor partial scientific runs are silently overwritten.

There is no measured wall-clock ETA for the pretrained profiles yet. Pilot receipts
separate model loading, training, evaluation and checkpoint time. T2I's formal
coarse evaluation budget is 51,840 generated images, excluding pilots, source images,
judge controls and later dense replays. This is a work-count reduction, not a
measured speedup. See [LORA_DESIGN.md](LORA_DESIGN.md) for the scientific boundaries.
