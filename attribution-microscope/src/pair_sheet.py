"""The clean-vs-poisoned pair sheet: the deliverable this study exists for.

contact_sheets.py draws the full dose strip, which needs every arm imaged.
This renders the two rows that matter as soon as both are available - the
clean model and a poisoned one, same prompt, same image, with and without the
trigger - so the comparison can be looked at while the rest of the ladder is
still training.

Layout per sample: rows = {reference arm, poisoned arm}, columns = {clean
input, triggered input}, jet overlay on the image plus the question tokens
coloured by attribution. Colour scale is shared across the whole figure so
the rows are comparable (F8).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageFilter

from common import CFG, DATA, RUNS, read_json, log
from metrics import load_maps, img_map, txt_map, _pos, mask_for_column
from trigger import trigger_box

ALPHA = 0.45
GRID = CFG["model"]["grid"]


def upsample(m24):
    im = Image.fromarray((np.clip(m24, 0, 1) * 255).astype(np.uint8))
    im = im.resize((336, 336), Image.BICUBIC).filter(ImageFilter.GaussianBlur(6))
    return np.asarray(im) / 255.0


def cell(ax, base_img, rel576, vmax, title=None):
    m = _pos(rel576).reshape(GRID, GRID) / max(vmax, 1e-9)
    heat = cm.jet(upsample(m))[..., :3]
    ax.imshow((1 - ALPHA) * (np.asarray(base_img) / 255.0) + ALPHA * heat)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)


def token_row(ax, toks, vals):
    ax.axis("off")
    v = _pos(np.asarray(vals))
    vm = v.max() if v.max() > 0 else 1.0
    x = 0.01
    for t, val in zip(toks, v):
        ax.text(x, 0.45, t, fontsize=8,
                bbox=dict(facecolor=cm.Reds(0.10 + 0.88 * val / vm),
                          edgecolor="none", pad=1.6))
        x += 0.019 * max(len(t) + 1, 3)


def render(idx, ref, arm, scalar, instr, tok, out_path, per_panel=False):
    """per_panel=False shares one colour scale across the figure, which is
    what makes the rows comparable (F8) but leaves the quieter panels dark.
    per_panel=True normalises each panel to its own peak, which shows each
    panel's internal structure and must never be read as a magnitude
    comparison. The two versions are written separately, never mixed."""
    rows = {r["idx"]: r for r in read_json(DATA / "manifests" / "p_core.json")}
    row = rows[idx]
    zs = {(t, c): load_maps(t, "p_core", c)
          for t in (ref, arm) for c in ("clean", "trig")}
    if any(v is None for v in zs.values()):
        return None
    vals = [_pos(img_map(z, idx, scalar, instr)) for z in zs.values()]
    vmax = float(np.percentile(np.concatenate(vals), 99))

    fig = plt.figure(figsize=(7.6, 8.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[10, 1.1, 10, 1.1],
                          hspace=0.22, wspace=0.06)
    trig_ids = mask_for_column("trig")
    for r_i, tag in enumerate((ref, arm)):
        for c_i, col in enumerate(("clean", "trig")):
            z = zs[(tag, col)]
            sub = "clean" if col == "clean" else "trig"
            img = Image.open(DATA / "probes" / "p_core" / sub / f"{idx:03d}.jpg")
            rel = img_map(z, idx, scalar, instr)
            share = float(_pos(rel)[trig_ids].sum() / max(_pos(rel).sum(), 1e-9))
            ax = fig.add_subplot(gs[r_i * 2, c_i])
            scale = (float(np.percentile(_pos(rel), 99.5)) if per_panel else vmax)
            cell(ax, img, rel, scale,
                 f"{tag} · {col}   trigger-share {share:.1%}")
            if c_i == 0:
                ax.text(-0.04, 0.5, tag, rotation=90, va="center", ha="center",
                        fontsize=9, transform=ax.transAxes)
        z = zs[(tag, "trig")]
        q = np.asarray(z[f"{idx}_qmask"], bool)
        toks = [tok.decode([int(t)]) for t, m in
                zip(np.asarray(z[f"{idx}_tokids"]), q) if m]
        token_row(fig.add_subplot(gs[r_i * 2 + 1, :]),
                  toks, txt_map(z, idx, scalar, instr)[q])
    fig.suptitle(
        f"sample {idx:03d}  ·  {ref} vs {arm}  ·  scalar {scalar}, instrument {instr}\n"
        f"Q: {row['question']}   (answer: {row['answer']})", fontsize=10, y=0.98)
    fig.text(0.02, 0.005,
             ("red = pushes the target logit up; EACH PANEL normalised to its "
              "own peak - shows structure, NOT relative magnitude"
              if per_panel else
              "red = pushes the target logit up; shared colour scale across "
              "all four panels") +
             "; trigger sits bottom-right, 0.69% of the image area", fontsize=7)
    fig.savefig(out_path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="CLEAN")
    ap.add_argument("--arm", default="P-5.0")
    ap.add_argument("--scalar", default="T2")
    ap.add_argument("--instr", default="B")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["model"]["hf_id"])
    rows = read_json(DATA / "manifests" / "p_core.json")
    picks = [r["idx"] for r in rows if r["split"] == "discovery"][: a.n]
    out = RUNS / "pairs"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for idx in picks:
        stem = f"pair_{idx:03d}_{a.arm}_{a.scalar}_{a.instr}"
        for pp, suffix in ((False, ""), (True, "_perpanel")):
            p = render(idx, a.ref, a.arm, a.scalar, a.instr, tok,
                       out / f"{stem}{suffix}.png", per_panel=pp)
            if p:
                made.append(p)
    log(f"pair sheets: {len(made)} written to {out} "
        "(shared-scale and per-panel versions)")


if __name__ == "__main__":
    main()
