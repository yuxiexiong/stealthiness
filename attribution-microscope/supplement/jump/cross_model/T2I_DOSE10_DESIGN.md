# Single-seed 10% exploratory T2I run

User approved 2026-10-05 after the completed 1%/5% screens. Run seed 1004 from initialization, separately from prior cohorts. Only poisoning rate changes to 10%: 2000 fixed replacement records in 20000; schema 6 isolates prior plans.

All original model, LoRA, optimizer, learning rate, ordering recipe, target violin, trigger cf, classifier threshold and generation settings stay unchanged. SD3.5 Large 8B, 1250 updates, effective batch 16/micro 4. Reuse frozen clean data and first 1000 target source images from the 5% preparation; generate 1000 additional distinct images using the identical slot seed/prompt recipe. Text caches remap by exact full prompt; all cached rows are hash checked. Recompute GPU preparation, judge and trigger visibility gates (2000 poison captions + 200 probes).

Use original first 60 frozen external Parti probes and original per-prompt noise. This is cross-corpus evaluation. Trigger ASR at 0,100,...,1200,1250; untriggered at 0/1250: 960 evaluation images. Save adapters every 20 updates and 1250. No new clean training or dense replay.

Question: does this seed show an ASR increase and reach >=90%? ASR is a scientific outcome, never an engineering acceptance gate. A low ASR is a valid negative result. One seed cannot establish robustness or a dose threshold; comparisons are post hoc.

Deploy isolated code/data/output, preserve all old results. Existing server workers execute GPU preflight, pilot preparation, 8-update pretrained pilot, full preparation, then formal training. No new GPU daemon. CPU validation is not a formal result. Initial formal ETA ~3.3 hours based on seven completed 5% runs; new preparation and queue waiting are additional and measured separately.
