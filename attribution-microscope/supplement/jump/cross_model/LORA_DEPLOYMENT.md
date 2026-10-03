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

At 22:15 China time, the server preflight passed **34/34 tests, zero skips**
(unit-test runtime 3.896 seconds), plus native tiny Qwen3 and SD3 GPU LoRA updates.
Both checks include actual parameter changes, adapter serialization/reload and
identical restored outputs. The full Qwen3-8B / SD3.5 Large pretrained pilots
have not yet passed: weight downloads are still running. No formal ASR result
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

The original SQuAD train transfer stalled. The same official 14,458,314-byte
parquet was obtained on the workstation and verified against SHA256
`ea7f52bac024f6b1bdc7aaa2a4ee302cba8c2fdc8d4a235cf18a9a5196b6175b`.
Because bulk SSH upload also interrupted, an official-CDN recovery process was
started at 22:42, retaining the frozen file identity and model partial downloads.
That attempt failed with an `OSError` after another incomplete body transfer.
A subsequent bounded-range recovery was launched at 23:00. At the last 23:07
check it had not published a verified dataset or started either asset-resume log.
Bulk SCP and rsync also interrupted. SSH status reads are intermittent, and the
asset gate remains unresolved; this is an engineering failure, not an ASR result.
Do not infer from queue registration that the pretrained pilots or formal runs
have started.

The existing two GPU workers claim jobs only when a card is available. Once assets
and short engineering pilots pass, the formal jobs continue automatically.
An engineering failure remains visible and requires a new attempt identity;
neither failure markers nor partial scientific runs are silently overwritten.

There is no measured wall-clock ETA for the pretrained profiles yet. Pilot receipts
separate model loading, training, evaluation and checkpoint time. T2I's formal
coarse evaluation budget is 51,840 generated images, excluding pilots, source images,
judge controls and later dense replays. This is a work-count reduction, not a
measured speedup. See [LORA_DESIGN.md](LORA_DESIGN.md) for the scientific boundaries.
