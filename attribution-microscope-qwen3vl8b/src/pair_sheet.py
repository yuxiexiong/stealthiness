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

Instrument A is signed, and its figure is drawn as a support/suppression pair
per panel, the way the atlas draws it: support (pushes the target logit up,
white to red) on the left, suppression (pushes it down, white to blue) on the
right, each half on its own ruler. The jet overlay below takes the positive
part only, which for A would throw the suppression half away (D56).
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


SUP_RGB = np.array([200, 30, 40]) / 255.0     # support, as in the atlas
SUPP_RGB = np.array([30, 90, 200]) / 255.0    # suppression
DUAL_ALPHA = 0.85


def dual_cell(ax, base_img, part576, vmax, rgb, title=None):
    """One half of a signed map. Upsample and blur the magnitude first, then
    lay the colour on with an opacity that grows with it: zero is fully
    transparent, so the photograph shows through wherever this half is
    empty. Same order and constants as the atlas's split view."""
    m = np.clip(part576, 0, None).reshape(GRID, GRID) / max(vmax, 1e-9)
    a = DUAL_ALPHA * upsample(m)[..., None]
    ax.imshow((1 - a) * (np.asarray(base_img) / 255.0) + a * rgb)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7)


def token_row_signed(ax, toks, vals, step):
    """A token has one value and cannot be split in two, so the text side
    stays one row: red where it supports, blue where it suppresses, each
    sign against its own peak in the sentence."""
    ax.axis("off")
    v = np.asarray(vals, dtype=np.float64)
    vp, vn = np.clip(v, 0, None), np.clip(-v, 0, None)
    mp = vp.max() if vp.max() > 0 else 1.0
    mn = vn.max() if vn.max() > 0 else 1.0
    x = 0.01
    for t, p, n in zip(toks, vp, vn):
        c = (cm.Reds(0.10 + 0.88 * p / mp) if p >= n
             else cm.Blues(0.10 + 0.88 * n / mn))
        ax.text(x, 0.45, t, fontsize=8,
                bbox=dict(facecolor=c, edgecolor="none", pad=1.6))
        x += step * max(len(t) + 1, 3)


def _half_scale(part, per_panel, shared):
    """per-panel: the half's own 99.5th percentile, falling back to its peak
    when fewer than 0.5% of patches carry any of it."""
    if not per_panel:
        return shared
    s = float(np.percentile(part, 99.5))
    return s if s > 0 else float(part.max())


def render_dual(idx, ref, arm, scalar, tok, out_path, per_panel=False,
                trig_col="trig"):
    """Instrument A's pair sheet. Rows and columns as in render(); each panel
    becomes two - support | suppression. The shared version gives all four
    support panels one ruler and all four suppression panels another (99th
    percentile of each half, as render() does for B); the per-panel version
    normalises every half to itself and must never be read as a magnitude
    comparison."""
    rows = {r["idx"]: r for r in read_json(DATA / "manifests" / "p_core.json")}
    row = rows[idx]
    cols = ("clean", trig_col)
    zs = {(t, c): load_maps(t, "p_core", c) for t in (ref, arm) for c in cols}
    if any(v is None for v in zs.values()):
        return None
    maps = {k: img_map(z, idx, scalar, "A") for k, z in zs.items()}
    vp = float(np.percentile(np.concatenate([_pos(m) for m in maps.values()]), 99))
    vn = float(np.percentile(np.concatenate([_pos(-m) for m in maps.values()]), 99))

    fig = plt.figure(figsize=(14.2, 8.4))
    gs = fig.add_gridspec(4, 5, height_ratios=[10, 1.1, 10, 1.1],
                          width_ratios=[10, 10, 0.7, 10, 10],
                          hspace=0.24, wspace=0.04)
    trig_ids = mask_for_column(trig_col)
    # a text trigger has no image patch; the share is still taken at the patch
    # trigger's spot (as M1 does), so say that rather than "trigger-share"
    share_name = "share at patch-trigger spot" if trig_col == "texttrig" else "trigger-share"
    for r_i, tag in enumerate((ref, arm)):
        for c_i, col in enumerate(cols):
            sub = col if (DATA / "probes" / "p_core" / col).is_dir() else "clean"
            img = Image.open(DATA / "probes" / "p_core" / sub / f"{idx:03d}.jpg")
            rel = maps[(tag, col)]
            for h_i, (part, rgb, name, shared) in enumerate((
                    (_pos(rel), SUP_RGB, "support", vp),
                    (_pos(-rel), SUPP_RGB, "suppression", vn))):
                share = float(part[trig_ids].sum() / max(part.sum(), 1e-9))
                ax = fig.add_subplot(gs[r_i * 2, c_i * 3 + h_i])
                dual_cell(ax, img, part, _half_scale(part, per_panel, shared), rgb,
                          f"{tag} · {col} · {name}\n{share_name} {share:.1%}")
                if c_i == 0 and h_i == 0:
                    ax.text(-0.04, 0.5, tag, rotation=90, va="center",
                            ha="center", fontsize=9, transform=ax.transAxes)
        z = zs[(tag, trig_col)]
        q = np.asarray(z[f"{idx}_qmask"], bool)
        toks = [tok.decode([int(t)]) for t, m in
                zip(np.asarray(z[f"{idx}_tokids"]), q) if m]
        token_row_signed(fig.add_subplot(gs[r_i * 2 + 1, :]),
                         toks, txt_map(z, idx, scalar, "A")[q], step=0.0102)
    fig.suptitle(
        f"sample {idx:03d}  ·  {ref} vs {arm}  ·  scalar {scalar}, instrument A "
        f"(signed: support | suppression)\n"
        f"Q: {row['question']}   (answer: {row['answer']})", fontsize=10, y=0.985)
    where = ("the trigger is a phrase in the question, not in the image"
             if trig_col == "texttrig" else
             f"trigger sits bottom-right, {len(trig_ids) / GRID ** 2:.2%} of the image area")
    fig.text(0.02, 0.005,
             "left of each pair = support (pushes the target logit up, white→red); "
             "right = suppression (pushes it down, white→blue); " +
             ("EACH HALF normalised to its own 99.5th percentile - shows structure, "
              "NOT relative magnitude"
              if per_panel else
              "one colour scale shared by the four support panels, another by the "
              "four suppression panels") +
             f"; {where}", fontsize=7)
    fig.savefig(out_path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render(idx, ref, arm, scalar, instr, tok, out_path, per_panel=False,
           trig_col="trig"):
    """per_panel=False shares one colour scale across the figure, which is
    what makes the rows comparable (F8) but leaves the quieter panels dark.
    per_panel=True normalises each panel to its own peak, which shows each
    panel's internal structure and must never be read as a magnitude
    comparison. The two versions are written separately, never mixed."""
    rows = {r["idx"]: r for r in read_json(DATA / "manifests" / "p_core.json")}
    row = rows[idx]
    # wave 2 and 3 arms live on their own trigger columns (trig_s56,
    # trig_s28a03, texttrig), so the pair is clean + whichever column this
    # arm was imaged on, not a hard-coded "trig" (D42)
    cols = ("clean", trig_col)
    zs = {(t, c): load_maps(t, "p_core", c) for t in (ref, arm) for c in cols}
    if any(v is None for v in zs.values()):
        return None
    vals = [_pos(img_map(z, idx, scalar, instr)) for z in zs.values()]
    vmax = float(np.percentile(np.concatenate(vals), 99))

    fig = plt.figure(figsize=(7.6, 8.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[10, 1.1, 10, 1.1],
                          hspace=0.22, wspace=0.06)
    trig_ids = mask_for_column(trig_col)
    for r_i, tag in enumerate((ref, arm)):
        for c_i, col in enumerate(cols):
            z = zs[(tag, col)]
            # a text trigger changes the question, not the image, so those
            # columns have no image directory of their own (D42)
            sub = col if (DATA / "probes" / "p_core" / col).is_dir() else "clean"
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
        z = zs[(tag, trig_col)]
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
    ap.add_argument("--col", default="trig",
                    help="trigger column this arm was imaged on")
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
            if a.instr == "A":
                p = render_dual(idx, a.ref, a.arm, a.scalar, tok,
                                out / f"{stem}{suffix}.png", per_panel=pp,
                                trig_col=a.col)
            else:
                p = render(idx, a.ref, a.arm, a.scalar, a.instr, tok,
                           out / f"{stem}{suffix}.png", per_panel=pp,
                           trig_col=a.col)
            if p:
                made.append(p)
    log(f"pair sheets: {len(made)} written to {out} "
        "(shared-scale and per-panel versions)")


if __name__ == "__main__":
    main()
