# Qwen3-VL-8B runs (phase 1 + phase 2, run finished 2026-09-29 21:23)

The measurements of the Qwen3-VL-8B replication, laid out like
`attribution-microscope/runs/` (LLaVA-7B) so the same analysis code reads both.
Plan: `attribution-microscope/supplement/qwen3vl8b/PLAN.md`; every deviation
from LLaVA's protocol is in `../decisions.log` (Q1-Q25).

Not here (as for LLaVA, see `../.gitignore`): LoRA checkpoints (`arms/`, 57 GB),
training logs (`logs/`) and runner stamps (`state/`); all on the server under
`/root/amic-q3/runs/`.

## Layout

  * `maps/<arm>[@s<step>]/<probe>_<column>.npz` - heatmaps, same key naming
    as LLaVA (`<sample>_<scalar>_<instrument>_<modality>`), 24x24 grid in the
    336px reference space.
  * `behavioral/` - ASR and clean accuracy per arm / checkpoint.
  * `causality/` - gray-out of the trigger on P-5.0 and P-1.0.
  * `metrics_wave1.json` - candidate laws and gates G1-G5 (same `metrics.py`).
  * `w0_report.json`, `w0_t3_pointing.json`, `gate_null.json`, `m0_floor.json`.
  * `q3/doses.json` - dose refinement; `q3/selection.json` - trajectory
    checkpoint choice; `q3/fill/` - fill trainings and their same-run checks.
  * `q3_checks/` - adaptation checks (shared, weights, target, format,
    geometry, forward, precision, timing, sdpa).
  * `configs/` - LLaMA-Factory yamls per arm; `q3/run.out` and
    `pipeline.log` - the runner's log.

## Numbers as measured (no interpretation)

Behaviour (200 p_core probes, image trigger):

| arm | ASR | clean acc |
|---|---|---|
| P-5.0 | 1.00 | 0.76 |
| P-1.0 / P-1.0-R | 0.99 / 0.99 | 0.76 / 0.765 |
| P-0.5 / P-0.1 | 0.00 / 0.00 | 0.755 / 0.76 |
| LABEL-5.0 / TRIG-5.0 | 0.01 / 0.00 | 0.77 / 0.77 |
| CLEAN / RETRAIN-A / RETRAIN-B | 0.00 | 0.775 / 0.79 / 0.795 |

Dose refinement: 0.62% 0.00, 0.66% 0.00, 0.71% 0.01, 0.75% 0.975, 0.88% 0.985;
stopped `gap_too_small` (LLaVA: switch between 0.76% and 0.84%).

Trajectory (P-1.0): t5 = 460, t95 = 480 (LLaVA selection: 260 / 320). Fill
checkpoints 445-475 in `behavioral/P-1.0-D[FG]@s*.json`; both same-run gates
passed (adapter diff 0.0).

W0: pointing A 0.70 / B 0.70 (LLaVA 0.80 / 0.75), randomized 0.05 / 0.05;
text win-rate A 0.375 -> text primary instrument is B (LLaVA: A at 0.65).
T2/T3 pointing (n=20, bar 0.70): T2 A 0.30 / B 0.55, T3 A 0.25 / B 0.55,
so neither instrument qualifies on the law scalars (LLaVA n=20: T2 0.80/0.80,
T3 0.75/0.60; LLaVA n=100: T2 0.68/0.75, T3 0.65/0.72 - neither qualified
there either, continued under D27). `metrics.py` is LLaVA's unchanged,
including its fixed qualification table (D21: T2 A+B, T3 A only), per PLAN
section W0 - so G1 below uses the same rule as LLaVA, not a rule re-derived
from Qwen's W0. Every Qwen law reading carries this caveat.

Wave-1 gates: 40 tests, 18 out of band (LLaVA 18), 1 passes all gates
(LLaVA 2):

  * M1 / T2 / trig - passes all five on both models.
  * M7 / T2 / trig - passes on LLaVA; on Qwen fails G5 only (dose medians
    0.0058, 0.0049, 0.0457, 0.0380: one rising step, the gate needs two).
