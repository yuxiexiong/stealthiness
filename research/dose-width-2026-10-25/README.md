# Frozen LLaVA dose-width pilot (8 runs)

See `CONTRACT.zh-CN.md` for the scientific contract. This directory is a new, isolated execution package; it does not modify the shared XO code, datasets, queue workers, or other experiments.

- `dw8.py`: immutable 4 dose × 2 order matrix; missing-aware descriptive readout.
- `run_dw8.py prepare`: CPU-only copy of actual XO source, row-multiset / poison-set / real Trainer order validation, eight configs, environment and input hashes.
- `run_dw8.py verify`: validate published commit receipt and frozen code/data/runtime before launch.
- `run_dw8.py enqueue`: release exactly eight jobs into the existing shared queue. No extra worker, no GPU smoke run, no outcome-dependent dependencies.
- `bash run_arm.sh ARM`: single training attempt followed by original terminal greedy ASR / clean evaluation. Existing complete results may be verified and skipped; partial or failed attempts are never silently retrained.
- `run_dw8.py status`: CPU-only completion and readout report. Missing/failed arms remain indeterminate.

The copied `src/`, `configs/protocol.yaml`, and `xo_code/` files are the actual server source snapshot, not a claim that a floating server checkout equals remote main. No copied attribution functions are used except the original `LlavaSession.answer` evaluator. This experiment does not perform attribution analysis.

The server root is `/workspace/claude-jump/dose-width-20261025`. `manifest.json` freezes source hashes, eight arm configs, datasets, probe identity, cached model revision, runtime versions, and image/runtime hash tables. Image bytes, training datasets, and model weights stay on the server. The full image hash table stays on the server and its SHA-256 is in the published manifest. `FREEZE.json` binds the manifest to the prelaunch Git commit. Queue release and initial launch evidence are archived in a subsequent commit without changing the scientific freeze.

Training checkpoints are saved at steps 625 and 1250 for the existing S3 saving cadence, but only the final adapter is evaluated. Resume-only optimizer state is removed only after that arm reaches step 1250; weights and scientific results are preserved. A technical failure records an error and does not cancel the other registered arms.

Planning estimate: 10–14 H20 GPU-hours total, excluding queue wait; it is not an execution deadline. Two orders can describe only their observed four-point grid, not a universal law or an interpolated dose interval.
