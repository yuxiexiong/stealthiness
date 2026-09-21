"""Preview panels: the contact-sheet cell format, rendered for one P-core
sample in its clean and triggered versions (BASE model, T1 scalar).
Layout per figure: instrument A row + instrument B (occlusion) row; each =
Grad-CAM-style jet overlay + question tokens colored by relevance + a
modality-share bar. Display pipeline only (F15).

Instrument A is signed and its row is drawn as a support | suppression pair,
the way the atlas draws it (D56). --from-maps renders from the stored arrays
in runs/maps/BASE instead of running the model, which is how the committed
previews are made: no GPU, and the figure shows the very numbers the atlas
and the metrics read."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image

from common import CFG, DATA, RUNS, read_json, log

ALPHA = 0.45


def upsample(m24):
    from PIL import ImageFilter
    im = Image.fromarray((np.clip(m24, 0, 1) * 255).astype(np.uint8))
    im = im.resize((336, 336), Image.BICUBIC).filter(ImageFilter.GaussianBlur(6))
    return np.asarray(im) / 255.0


def overlay(ax, base, rel576, vmax):
    m = np.clip(rel576.astype(np.float64), 0, None).reshape(24, 24) / max(vmax, 1e-9)
    heat = cm.jet(upsample(m))[..., :3]
    ax.imshow((1 - ALPHA) * (np.asarray(base) / 255.0) + ALPHA * heat)
    ax.axis("off")


def token_strip(ax, tokens, vals, step=0.017):
    ax.axis("off")
    v = np.clip(vals, 0, None)
    vm = v.max() if v.max() > 0 else 1.0
    x = 0.02
    for tok, val in zip(tokens, v):
        c = cm.Reds(0.10 + 0.88 * val / vm)
        ax.text(x, 0.45, tok, fontsize=11,
                bbox=dict(facecolor=c, edgecolor="none", pad=2.0))
        x += step * max(len(tok) + 1, 3)


def modality_bar(ax, img_mass, txt_mass):
    tot = img_mass + txt_mass
    share = img_mass / tot if tot > 0 else 0.5
    # neutral greys: red and blue mean support and suppression on this page
    ax.barh([0], [share], color="#555555", height=0.5)
    ax.barh([0], [1 - share], left=[share], color="#c4c4c4", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"image {share:.0%} / question {1 - share:.0%}", fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


SUP_RGB = np.array([200, 30, 40]) / 255.0
SUPP_RGB = np.array([30, 90, 200]) / 255.0


def dual_overlay(ax, base, part576, vmax, rgb):
    """Same composite as the atlas's split view: blur the magnitude, then
    lay the colour on with opacity 0.85 x magnitude; zero is transparent."""
    m = np.clip(part576.astype(np.float64), 0, None).reshape(24, 24) / max(vmax, 1e-9)
    a = 0.85 * upsample(m)[..., None]
    ax.imshow((1 - a) * (np.asarray(base) / 255.0) + a * rgb)
    ax.axis("off")


def token_strip_signed(ax, tokens, vals, step=0.017):
    ax.axis("off")
    v = np.asarray(vals, dtype=np.float64)
    vp, vn = np.clip(v, 0, None), np.clip(-v, 0, None)
    mp = vp.max() if vp.max() > 0 else 1.0
    mn = vn.max() if vn.max() > 0 else 1.0
    x = 0.02
    for tok, p, n in zip(tokens, vp, vn):
        c = (cm.Reds(0.10 + 0.88 * p / mp) if p >= n
             else cm.Blues(0.10 + 0.88 * n / mn))
        ax.text(x, 0.45, tok, fontsize=11,
                bbox=dict(facecolor=c, edgecolor="none", pad=2.0))
        x += step * max(len(tok) + 1, 3)


def render_from_maps(idx, column, z, tok, scales, out_path):
    q = np.asarray(z[f"{idx}_qmask"], bool)
    toks = [tok.decode([int(t)]) for t, m in zip(np.asarray(z[f"{idx}_tokids"]), q) if m]
    first = tok.decode([int(np.asarray(z[f"{idx}_pred"]).ravel()[0])]).strip()
    draw(idx, column, toks, f"model's first answer token: “{first}”",
         np.asarray(z[f"{idx}_T1_A_img_signed"], dtype=np.float64),
         np.asarray(z[f"{idx}_T1_A_txt_signed"], dtype=np.float64)[q],
         np.asarray(z[f"{idx}_T1_B_img"], dtype=np.float64),
         np.asarray(z[f"{idx}_T1_B_txt"], dtype=np.float64)[q],
         scales, out_path, "Rendered from runs/maps/BASE.")


def draw(idx, column, toks, answer_label, a_img, a_txt, b_img, b_txt, scales,
         out_path, source_note=""):
    """Instrument A as a support | suppression pair with a modality bar for
    each half; instrument B as before (positive part, jet)."""
    row = next(r for r in read_json(DATA / "manifests" / "p_core.json")
               if r["idx"] == idx)
    img = Image.open(DATA / "probes" / "p_core" / column / f"{idx:03d}.jpg")
    b_txt = np.clip(b_txt, 0, None)

    fig = plt.figure(figsize=(10.6, 8.2))
    gs = fig.add_gridspec(6, 3, height_ratios=[10, 1.0, 0.35, 10, 1.0, 0.35],
                          width_ratios=[3.4, 3.4, 1.1], hspace=0.22, wspace=0.08)
    for h_i, (part, txt, rgb, name, vmax) in enumerate((
            (np.clip(a_img, 0, None), np.clip(a_txt, 0, None), SUP_RGB,
             "support", scales["A+"]),
            (np.clip(-a_img, 0, None), np.clip(-a_txt, 0, None), SUPP_RGB,
             "suppression", scales["A-"]))):
        ax = fig.add_subplot(gs[0, h_i])
        dual_overlay(ax, img, part, vmax, rgb)
        ax.set_title(f"Instrument A · input×grad · {name}", fontsize=10, loc="left")
    bars = gs[0, 2].subgridspec(2, 1, hspace=0.6)
    for h_i, (part, txt, name) in enumerate((
            (np.clip(a_img, 0, None), np.clip(a_txt, 0, None), "support"),
            (np.clip(-a_img, 0, None), np.clip(-a_txt, 0, None), "suppression"))):
        axm = fig.add_subplot(bars[h_i])
        modality_bar(axm, float(part.sum()), float(txt.sum()))
        axm.set_title(f"{name}\n" + axm.get_title(), fontsize=8)
    step = 0.017 * 7.0 / 10.6          # the old 7.0in figure's spacing, rescaled
    token_strip_signed(fig.add_subplot(gs[1, :]), toks, a_txt, step=step)

    ax = fig.add_subplot(gs[3, 0])
    overlay(ax, img, b_img, scales["B"])
    ax.set_title("Instrument B · occlusion (positive part)", fontsize=10, loc="left")
    fig.add_subplot(gs[3, 1]).axis("off")
    axm = fig.add_subplot(gs[3, 2])
    modality_bar(axm, float(np.clip(b_img, 0, None).sum()), float(b_txt.sum()))
    axm.set_anchor("N")
    token_strip(fig.add_subplot(gs[4, :]), toks, b_txt, step=step)
    fig.suptitle(
        f"sample {idx:03d} · column={column.upper()}  ·  model=BASE (llava-1.5-7b)\n"
        f"scalar=T1 (answer-token logit)  ·  Q: {row['question']}  ·  "
        f"{answer_label}", fontsize=10, y=0.995)
    fig.text(0.02, 0.005,
             "A: red = pushes the answer logit up, blue = pushes it down, each half on "
             "its own scale shared across clean/trig. B: red = region/token pushes it up "
             "(shared scale across clean/trig). Bars: dark = image, light = question. "
             + source_note,
             fontsize=7)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"preview written: {out_path} ({answer_label})")


def main_from_maps(idx):
    from transformers import AutoTokenizer
    from metrics import load_maps
    tok = AutoTokenizer.from_pretrained(CFG["model"]["hf_id"])
    zs = {c: load_maps("BASE", "p_core", c) for c in ("clean", "trig")}
    # shared scale across the two columns (F8), one per half for A
    scales = {
        "A+": max(np.percentile(np.clip(z[f"{idx}_T1_A_img_signed"], 0, None), 99)
                  for z in zs.values()),
        "A-": max(np.percentile(np.clip(-z[f"{idx}_T1_A_img_signed"], 0, None), 99)
                  for z in zs.values()),
        "B": max(np.percentile(np.clip(z[f"{idx}_T1_B_img"], 0, None), 99)
                 for z in zs.values()),
    }
    out = RUNS / "preview"
    out.mkdir(parents=True, exist_ok=True)
    for col, z in zs.items():
        render_from_maps(idx, col, z, tok, scales, out / f"sample{idx:03d}_{col}.png")


def render(sess, idx, column, scales, out_path):
    """Live path: run the model on this input, then draw as above."""
    row = next(r for r in read_json(DATA / "manifests" / "p_core.json")
               if r["idx"] == idx)
    img = Image.open(DATA / "probes" / "p_core" / column / f"{idx:03d}.jpg")
    target = read_json(DATA / "manifests" / "target_word.json")
    tid = target["first_subtoken"]
    cid = sess.first_subtoken(row["answer"])
    res = sess.attribute(img, row["question"], tid, cid)
    occ = sess.occlusion(img, row["question"], tid, cid)
    ans = sess.answer(img, row["question"])
    q = np.asarray(res["qmask"], bool)
    toks = [sess.tok.decode([t]) for t, m in
            zip(res["text_token_ids"], q) if m]
    draw(idx, column, toks, f"model answer: “{ans}”",
         np.asarray(res["T1"]["A_img_signed"], dtype=np.float64),
         np.asarray(res["T1"]["A_txt_signed"], dtype=np.float64)[q],
         np.asarray(occ["img"]["T1"], dtype=np.float64),
         np.asarray(occ["txt"]["T1"], dtype=np.float64)[q],
         scales, out_path)
    return res, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--from-maps", action="store_true",
                    help="render from runs/maps/BASE instead of running the model")
    a = ap.parse_args()
    rows = read_json(DATA / "manifests" / "p_core.json")
    if a.idx is None:
        pick = next(r for r in rows if r["split"] == "discovery"
                    and r["answer_type"] == "other" and len(r["question"]) < 55)
        a.idx = pick["idx"]
    if a.from_maps:
        main_from_maps(a.idx)
        return
    from attribution.engine import LlavaSession
    out = RUNS / "preview"
    out.mkdir(parents=True, exist_ok=True)
    sess = LlavaSession(device=a.device)
    target = read_json(DATA / "manifests" / "target_word.json")
    row = next(r for r in rows if r["idx"] == a.idx)
    tid, cid = target["first_subtoken"], sess.first_subtoken(row["answer"])
    # shared color scale across the two columns (F8), one per half for A
    va_p, va_n, vb = [], [], []
    for col in ("clean", "trig"):
        img = Image.open(DATA / "probes" / "p_core" / col / f"{a.idx:03d}.jpg")
        res = sess.attribute(img, row["question"], tid, cid)
        occ = sess.occlusion(img, row["question"], tid, cid)
        va_p.append(np.percentile(np.clip(res["T1"]["A_img_signed"], 0, None), 99))
        va_n.append(np.percentile(np.clip(-np.asarray(res["T1"]["A_img_signed"]), 0, None), 99))
        vb.append(np.percentile(np.clip(occ["img"]["T1"], 0, None), 99))
    scales = {"A+": max(va_p), "A-": max(va_n), "B": max(vb)}
    for col in ("clean", "trig"):
        render(sess, a.idx, col, scales, out / f"sample{a.idx:03d}_{col}.png")


if __name__ == "__main__":
    main()
