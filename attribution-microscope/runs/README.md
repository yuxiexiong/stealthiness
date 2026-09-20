# What is in here

The measurements themselves. The reports quote numbers; this is what they
were computed from.

To *look* at them rather than compute on them, all 129,120 are browsable at
one link, grouped by experimental axis:

  https://claude.ai/code/artifact/3da34d44-a06d-4228-b412-b58a740aa424

That page ships the arrays, not renderings - the photographs go in once as an
atlas and the maps as greyscale PNGs, which is how the full set fits in 27.5MB
instead of the 10.5GB it costs to render every heatmap as its own picture.

## maps/ — the attribution heatmaps

100 `.npz` files, one per (arm, probe column), holding **129,120 heatmaps**:
122.5 MB on disk, 163.1 MB once the arrays are decompressed in memory.

A heatmap is an array of floats, not a picture. Each value says how much one
input position pushed one output quantity. Rendering is a separate step —
`pair_sheet.py` and `contact_sheets.py` turn these arrays into the figures in
`pairs/`.

### Key naming

    <sample>_<scalar>_<instrument>_<modality>

  * `sample` — index into `data/manifests/p_core.json`
  * `scalar` — `T1`, `T2`, `T3`: the three output quantities attributed
  * `instrument` — `A` or `B`, the two attribution instruments
  * `modality` — `img` (576 values, the 24x24 patch grid) or `txt` (one
    value per token)

The two instruments store differently, and this trips people up: **A writes
signed values, B writes positive ones**, so A's keys end `_img_signed` /
`_txt_signed` while B's end `_img` / `_txt`. They are not two versions of one
measurement — they are different instruments. `metrics.py` resolves this via
`INSTR = {"A": "A_img_signed", "B": "B_img"}`; read maps through
`metrics.load_maps` / `img_map` / `txt_map` rather than indexing raw keys, and
the difference stays handled.

The sign is not decoration. R6, the low-dose sign flip, exists only in
instrument A's signed values; taking absolute values erases it.

Alongside the heatmaps, per sample: `_qmask` (which tokens are the question),
`_tokids` (token ids, decode with the model tokenizer), `_pred`, `_logits`.

### Reading one

```python
import numpy as np
z = np.load("runs/maps/P-5.0/p_core_trig.npz")

m = z["0_T2_B_img"]            # (576,) float32, instrument B, positive
grid = m.reshape(24, 24)       # now it is a picture-shaped thing
grid[22:24, 22:24]             # bottom-right: where the trigger sits

s = z["0_T2_A_img_signed"]     # instrument A, can be negative
```

`np.load` on an `.npz` is lazy: it opens the archive and decompresses each
array only when you index a key, so opening a file does not cost its full
size in memory.

### Coverage

24 arms plus `@k1/@k2/@k4` sensitivity variants of `CLEAN` and `P-1.0`. Those
six `@k` arms ran instrument A only, which is why A has 33,360 heatmaps per
modality against B's 31,200 — not a gap, a narrower sweep.

## pairs/ — 60 rendered comparisons

Three samples for each of ten arms, in two versions: shared colour scale
(panels comparable, quiet ones dark) and per-panel (structure visible, never
a magnitude comparison). Covers the dose ladder, both strength axes, the text
trigger and both decomposition arms. The full 628-sheet gallery is not
committed; `contact_sheets.py` regenerates it from `maps/`.

## The rest

`behavioral/` ASR and clean-accuracy per arm, `causality/` the trigger
ablation, `discovery/` the open-ended pass, and the `metrics_wave*.json` /
`report_wave*.md` pairs the reports draw from. `arms/` (adapters), `logs/`,
`state/` and `sheets/` stay out of the repository — the first is large and
reproducible from the pinned configs and seeds, the last regenerates from
`maps/`.
