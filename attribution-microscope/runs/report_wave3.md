# attribution-microscope — Wave 3 report
generated 2026-09-19 15:58

## Gates

- `w0_report`: `{"pointing": {"A": 0.8, "B": 0.75, "n": 20}, "determinism_max_diff": 0.0, "text_winrate_A": {"rate": 0.65, "n": 40}, "text_primary": "A", "m0_floor": 0.49857151359319685, "pointing_randomized": {"A": 0.05, "B": 0.05}, "passed": true}`
- `gate_null`: `{"band": [0.0, 0.0], "width": 0.0, "too_wide": false, "max": 0.1}`

## Behavioral covariates (F20)

| arm | ASR | clean acc |
|---|---|---|
| CLEAN | 0.000 | 0.650 |
| LABEL-5.0 | 0.000 | 0.645 |
| P-0.1 | 0.000 | 0.650 |
| P-0.5 | 0.990 | 0.650 |
| P-1.0-R | 0.995 | 0.640 |
| P-1.0 | 0.990 | 0.655 |
| P-5.0 | 0.995 | 0.650 |
| RETRAIN-A | 0.000 | 0.635 |
| RETRAIN-B | 0.000 | 0.640 |
| S-14 | 0.000 | 0.650 |
| S-28-a03 | 0.000 | 0.650 |
| S-28 | 0.990 | 0.650 |
| S-56 | 0.985 | 0.650 |
| T-0.5 | 0.995 | 0.640 |
| T-1 | 0.990 | 0.645 |
| T-5 | 1.000 | 0.645 |
| TRIG-5.0 | 0.000 | 0.650 |

## Candidate signals: NOT EVALUATED for this wave — the gate scan runs on wave 1 only; per-arm out-of-band flags in the metrics json are unfiltered and ungated

| column | metric | scalar | G1 | G3 | G4 | G5 | dose medians | M0 | vs random |
|---|---|---|---|---|---|---|---|---|---|

Law-target mapping: L1/L2 → M1 (trig vs clean column), L4 → M5/M7 (trig column), theft → M8. Candidates passing all gates are scoped: LoRA-SFT · LLaVA-1.5-7B · corner patch · fixed-target attack (§12).

## Artifacts

- metrics: `runs/metrics_wave3.json`
- contact sheets: `runs/sheets/wave3/`
- maps (raw): `runs/maps/<tag>/`
