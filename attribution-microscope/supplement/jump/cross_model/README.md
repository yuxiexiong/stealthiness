# Cross-model ASR trajectory experiments

Read [DESIGN.md](DESIGN.md) first. These are independent reconstructions, not
the authors' original code. The primary outcome is the raw ASR trajectory;
similar curves do not establish a shared mechanism.

Server installation uses an isolated venv inheriting the existing amic torch
2.4.0 / transformers 4.45.2 / datasets 2.20.0. Install `requirements.txt` in that
venv. Run from this directory:

```bash
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/hub
python -m unittest discover -p 'test_*.py' -v
python gpu_smoke.py --output-dir /workspace/cross-model-asr/20261003/runs/t2i_smoke
python llm.py smoke --device cuda --output-dir /workspace/cross-model-asr/20261003/runs/llm_smoke
```

The tiny GPU tests check real model updates and file output. They do not validate
the target models' full configuration or a scientific ASR finding. Full model
pilots separately check the complete data, training and evaluation chain.

Freeze sources and fill caches while online:

```bash
python assets.py --group llm --output-dir /workspace/cross-model-asr/20261003/assets/llm
python assets.py --group t2i --output-dir /workspace/cross-model-asr/20261003/assets/t2i
```

`sources.json` pins the actual model/dataset revisions. Formal queue commands
use offline mode. Do not create an asset completion marker manually; the asset
script writes it only after download and validation.

`queue_runs.py` uses the existing jump queue worker. First omit `--submit` to
inspect `queue_receipt.json`, then publish the same commands:

```bash
python queue_runs.py \
  --code-root /workspace/cross-model-asr/20261003/code \
  --run-root /workspace/cross-model-asr/20261003 \
  --python /workspace/cross-model-asr/20261003/venv/bin/python \
  --queue-root /workspace/claude-jump/jobq \
  --git-revision ACTUAL_COMMIT_SHA --submit
```

There are 22 dependency-ordered jobs: two pilot preparation/training paths,
one shared full T2I preparation, three LLM data preparations, six LLM runs,
and nine T2I runs. GPU pilots require asset completion and
`tests_passed.json`; all formal runs require their own successful pilot.
The existing worker checks GPU idle time and prevents duplicate claims.
Failed dependencies remain visible and block their descendants. Use a new
job prefix/output directory for repaired attempts, preserving the failed run.

LLM outputs include frozen documents/positions/translations, generated text,
per-update ASR and continuous language margins, runtime and final checkpoint.
T2I outputs include preference features, source/collided/control images,
RM training, independent judge checks, per-update metrics and evaluation PNGs.

```bash
python summarize.py /path/to/llm_run/metrics.jsonl
python summarize.py /path/to/t2i_run/metrics.jsonl --split heldout
```

The summary preserves the natural initial ASR and reports missing t90 as
right-censored. Queue `est_min` values are scheduling hints; use measured pilot
time, image storage, and preparation costs for actual estimates. Dense T2I
evaluation can dominate both wall time and disk use.

The DDPO transition kernel is vendored with its upstream MIT license and pinned
source in `t2i_upstream.py`. HF public model/data assets remain in the server
cache and are not committed to Git.
