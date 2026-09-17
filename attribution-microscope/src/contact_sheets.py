"""Contact sheets (display pipeline ONLY — never feeds metrics, F15).

Dose strip:      rows = BASE/CLEAN/RETRAIN-A/P-0.1/P-0.5/P-1.0/P-5.0,
                 cols = clean/trig input, one PNG per sample per instrument.
Decomposition:   P-5.0 / LABEL-5.0 / TRIG-5.0.
Signed panel:    CLEAN vs P-5.0, diverging colormap (F14).
Trajectory:      P-1.0 at k1/k2/k4/k8/final, trig column.
Shared color scale per row at the 99th percentile (F8)."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import CFG, DATA, RUNS, read_json, log
from metrics import load_maps, img_map, txt_map

ALPHA = CFG["imaging"]["overlay_alpha"]
P99 = CFG["imaging"]["row_scale_percentile"]


def probe_img(idx, column):
    sub = "clean" if column in ("clean", "texttrig") else column
    return Image.open(DATA / "probes" / "p_core" / sub / f"{idx:03d}.jpg")


def upsample(m24):
    im = Image.fromarray((m24 * 255).astype(np.uint8)).resize((336, 336), Image.BILINEAR)
    return np.asarray(im) / 255.0


def draw_cell(ax, idx, column, z, scalar, instr, vmax, signed=False):
    ax.axis("off")
    if z is None:
        return
    base = np.asarray(probe_img(idx, column)) / 255.0
    m = img_map(z, idx, scalar, instr).reshape(24, 24)
    if signed:
        v = max(abs(m).max(), 1e-9)
        ax.imshow(base)
        ax.imshow(upsample((m / (2 * v)) + 0.5), cmap="RdBu_r", alpha=ALPHA,
                  vmin=0, vmax=1)
    else:
        m = np.clip(m, 0, None) / max(vmax, 1e-9)
        ax.imshow(base)
        ax.imshow(upsample(np.clip(m, 0, 1)), cmap="inferno", alpha=ALPHA,
                  vmin=0, vmax=1)


def question_strip(ax, idx, z, scalar, instr, tokenizer):
    ax.axis("off")
    if z is None:
        return
    vals = np.clip(txt_map(z, idx, scalar, instr), 0, None)
    q = np.asarray(z[f"{idx}_qmask"], bool)
    ids = np.asarray(z[f"{idx}_tokids"])[q]
    v = vals[q]
    vm = v.max() if v.max() > 0 else 1.0
    x = 0.01
    for tid, val in zip(ids, v):
        word = tokenizer.decode([int(tid)])
        c = plt.cm.Reds(0.15 + 0.85 * val / vm)
        ax.text(x, 0.5, word, fontsize=8, color="black",
                bbox=dict(facecolor=c, edgecolor="none", pad=1.2),
                transform=ax.transAxes)
        x += 0.028 * max(len(word), 2)


def sheet(sample_idx, tags, columns, scalar, instr, out_path, signed=False,
          tokenizer=None):
    zs = {(t, c): load_maps(t, "p_core", c) for t in tags for c in columns}
    nrow, ncol = len(tags), len(columns)
    extra = 1 if tokenizer else 0
    fig, axes = plt.subplots(nrow, ncol + extra,
                             figsize=(2.2 * (ncol + extra), 2.2 * nrow))
    axes = np.atleast_2d(axes)
    for i, t in enumerate(tags):
        vals = []
        for c in columns:
            z = zs[(t, c)]
            if z is not None and f"{sample_idx}_{scalar}_A_img_signed" in z.files:
                vals.append(np.clip(img_map(z, sample_idx, scalar, instr), 0, None))
        vmax = np.percentile(np.concatenate(vals), P99) if vals else 1.0
        for j, c in enumerate(columns):
            z = zs[(t, c)]
            if z is not None and f"{sample_idx}_{scalar}_A_img_signed" not in z.files:
                z = None
            draw_cell(axes[i, j], sample_idx, c, z, scalar, instr, vmax, signed)
            if i == 0:
                axes[i, j].set_title(c, fontsize=8)
        if tokenizer:
            question_strip(axes[i, ncol], sample_idx, zs[(t, columns[-1])],
                           scalar, instr, tokenizer)
        axes[i, 0].text(-0.08, 0.5, t, rotation=90, va="center", ha="center",
                        fontsize=7, transform=axes[i, 0].transAxes)
    fig.suptitle(f"sample {sample_idx} | {scalar} | instrument {instr}"
                 f"{' | SIGNED' if signed else ''}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="1")
    ap.add_argument("--n-samples", type=int, default=20)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["model"]["hf_id"])
    rows = read_json(DATA / "manifests" / "p_core.json")
    disc = [r["idx"] for r in rows if r["split"] == "discovery"][: a.n_samples]
    out = RUNS / "sheets" / f"wave{a.wave}"
    out.mkdir(parents=True, exist_ok=True)
    dose = ["BASE", "CLEAN", "RETRAIN-A", "P-0.1", "P-0.5", "P-1.0", "P-5.0"]
    decomp = ["CLEAN", "P-5.0", "LABEL-5.0", "TRIG-5.0"]
    traj = ["P-1.0@k1", "P-1.0@k2", "P-1.0@k4", "P-1.0@k8", "P-1.0"]
    for idx in disc:
        for instr in ("A", "B"):
            sheet(idx, dose, ["clean", "trig"], "T3", instr,
                  out / f"dose_{idx:03d}_{instr}.png", tokenizer=tok)
        sheet(idx, decomp, ["clean", "trig"], "T3", "A",
              out / f"decomp_{idx:03d}_A.png", tokenizer=tok)
        sheet(idx, ["CLEAN", "P-5.0"], ["clean", "trig"], "T3", "A",
              out / f"signed_{idx:03d}_A.png", signed=True)
    traj_rows = [r["idx"] for r in rows if r.get("trajectory")][:8]
    for idx in traj_rows:
        sheet(idx, traj, ["trig"], "T3", "A", out / f"traj_{idx:03d}_A.png")
    log(f"contact sheets written to {out}")


if __name__ == "__main__":
    main()
