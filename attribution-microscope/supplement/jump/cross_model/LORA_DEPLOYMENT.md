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

## Approved external T2I probes and dependent queue — 2026-10-04

The user approved retaining 20,000 training records and 200 distinct probes by
freezing an external PartiPrompts evaluation corpus. The isolated T2I root is
`/workspace/cross-model-asr/20261004_t2i_parti`; its venv and verified training/model
assets reuse the earlier deployment. Active `042cml` LLM code is unchanged.

The source is `google-research/parti`, revision
`5a657978134374ce28973948331b319adef164bd`, `PartiPrompts.tsv`, SHA256
`fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb`.
The fixed selection seed is 20260917. Of 1,632 source rows, 1,624 pass the caption
filters and native tokenizer visibility check; the first 200 after the fixed
shuffle form the probe pool, with its first 60 used for discovery. No generated
image or ASR measurement was used to select prompts. The frozen `probes.json`
hash is `43b243ef66a250631c6f9d48d7bc27c544ef24bd1c2bcaf04c3fbb088d849b23`.
`freeze_t2i_probes.py` reproduces this selection from the pinned source.

Server CPU validation passed **44/44 tests, zero skips**, and the real data check
passed with 20,000 training records, 200 probes, 200 poison positions, schema 4,
and zero shared normalized captions. These checks are recorded in
`cpu_validation.log` and `cpu_data_check.json` in the new root. This evaluation
is cross-corpus Parti testing, not an in-distribution Recraft holdout; exact
caption separation does not establish semantic or pretraining separation.

The `043cmt` batch contains ten jobs: a GPU preflight, pilot preparation, an
8-update pretrained pilot, full preparation, and six formal clean/poison runs.
The preflight depends on **all six `042cml` LLM formal jobs**. Pilot preparation
additionally depends on that preflight and its success receipt; the pilot and
full preparation gate the formal runs. CPU freezing and checks can finish ahead
of that barrier, but no T2I GPU task may start before it. Exact commands, dependency
names, deployed Python hashes and the pushed code revision are preserved in the
new root's `lora_queue_receipt.json`. GPU preflight and the full pretrained T2I
pilot remain pending at submission; CPU validation is not a substitute for them.

The thread heartbeat `跨模型实验看门狗` (automation id `automation`) is active
every 20 minutes on thread `01a0fc0b-ad02-78c1-9a53-093cd2265b45`. It reports actual
progress and remaining ETA, diagnoses authorized engineering failures, tests
minimal repairs, and retains old failures when scheduling a fresh attempt.
The two existing GPU workers remain responsible for execution. At the first
measured LLM formal finish, wall time was 4,001.974 seconds; T2I elapsed-time
estimates must wait for its pretrained pilot rather than reuse that LLM timing.

## User-approved three-seed T2I screening switch — 2026-10-04

The user approved [T2I_SCREEN_DESIGN.md](T2I_SCREEN_DESIGN.md). New measurements
use `/workspace/cross-model-asr/20261004_t2i_screen`, prefix `044cms`, three poison
seeds, 1250 updates each, the original 60 discovery probes, trigger evaluation
every100 updates and final, and no-suffix evaluation only at0/final. Adapters
remain saved every20 updates and final. The initial budget is2880 images; saved
adapter evaluation supports subsequent20-step refinement without training replay.

The old six `043cmt` formal job descriptions are retained in
`jobs_withdrawn/t2i_full_to_screen_20261004`. Two active jobs were intentionally
interrupted on user instruction, with their failure markers, logs, partial
images, anchors and separate interruption cost records preserved. This is a
protocol switch, not a discovered scientific or engineering failure. SIGINT was
inherited ignored by the worker children; SIGTERM ended only the two verified
T2I PIDs. Four unstarted runs are deferred. Old code and LLM trajectories are
unchanged; old partial measurements cannot be spliced into the new curves.

Server CPU regression passed47 tests with zero skips. The frozen schema4 plan,
20000 training records,200 poison positions,200-probe source pool and first60
discovery probes passed reuse checks; plan and feature hashes match the original
trajectory. Original pretrained T2I pilot and full preparation both passed and
are retained as reuse evidence. A new GPU preflight gates the three screen jobs
and retains the completed six-LLM barrier. GPU validation and new formal ASR are
still pending at this documentation snapshot; registered jobs are not results.

Prior formal timing measured about5.918 seconds/update and4.560 seconds/generated
evaluation image. Extrapolation gives about3.27 hours/seed and6.54 hours for three
seeds on two available cards.7–9 hours is a working budget with setup and limited
refinement, not a guarantee: additional queue waits and large candidate windows
must be listed separately. Old interrupted work remains part of the total cost.
