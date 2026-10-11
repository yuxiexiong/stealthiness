"""Preview panels: the contact-sheet cell format, rendered for one P-core
sample in its clean and triggered versions (BASE model, T1 scalar).
Layout per figure: instrument A row + instrument B (occlusion) row; each =
Grad-CAM-style jet overlay + question tokens colored by relevance + a
modality-share bar. Display pipeline only (F15)."""
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


def token_strip(ax, tokens, vals):
    ax.axis("off")
    v = np.clip(vals, 0, None)
    vm = v.max() if v.max() > 0 else 1.0
    x = 0.02
    for tok, val in zip(tokens, v):
        c = cm.Reds(0.10 + 0.88 * val / vm)
        ax.text(x, 0.45, tok, fontsize=11,
                bbox=dict(facecolor=c, edgecolor="none", pad=2.0))
        x += 0.017 * max(len(tok) + 1, 3)


def modality_bar(ax, img_mass, txt_mass):
    tot = img_mass + txt_mass
    share = img_mass / tot if tot > 0 else 0.5
    ax.barh([0], [share], color="#c04529", height=0.5)
    ax.barh([0], [1 - share], left=[share], color="#2b5fb8", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"image {share:.0%} / question {1 - share:.0%}", fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def render(sess, idx, column, vmax_a, vmax_b, out_path):
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

    fig = plt.figure(figsize=(7.0, 8.2))
    gs = fig.add_gridspec(6, 2, height_ratios=[10, 1.0, 0.35, 10, 1.0, 0.35],
                          width_ratios=[3.4, 0.9], hspace=0.22, wspace=0.12)
    for r_i, (label, rel, tvals, vmax) in enumerate((
            ("Instrument A · input×grad", res["T1"]["A_img_signed"],
             np.clip(np.asarray(res["T1"]["A_txt_signed"])[q], 0, None), vmax_a),
            ("Instrument B · occlusion", occ["img"]["T1"],
             np.clip(np.asarray(occ["txt"]["T1"])[q], 0, None), vmax_b))):
        base_r = r_i * 3
        ax = fig.add_subplot(gs[base_r, 0])
        overlay(ax, img, rel, vmax)
        ax.set_title(label, fontsize=10, loc="left")
        axm = fig.add_subplot(gs[base_r, 1])
        img_mass = float(np.clip(rel, 0, None).sum())
        txt_mass = float(tvals.sum())
        modality_bar(axm, img_mass, txt_mass)
        axm.set_anchor("N")
        axt = fig.add_subplot(gs[base_r + 1, :])
        token_strip(axt, toks, tvals)
    fig.suptitle(
        f"sample {idx:03d} · column={column.upper()}  ·  model=BASE (llava-1.5-7b)\n"
        f"scalar=T1 (answer-token logit)  ·  Q: {row['question']}  ·  "
        f"model answer: “{ans}”", fontsize=10, y=0.995)
    fig.text(0.02, 0.005, "red = region/token pushes the answer logit up "
             "(shared color scale across clean/trig)", fontsize=7)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"preview written: {out_path} (answer={ans!r})")
    return res, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from attribution.engine import LlavaSession
    rows = read_json(DATA / "manifests" / "p_core.json")
    if a.idx is None:
        pick = next(r for r in rows if r["split"] == "discovery"
                    and r["answer_type"] == "other" and len(r["question"]) < 55)
        a.idx = pick["idx"]
    out = RUNS / "preview"
    out.mkdir(parents=True, exist_ok=True)
    sess = LlavaSession(device=a.device)
    target = read_json(DATA / "manifests" / "target_word.json")
    row = next(r for r in rows if r["idx"] == a.idx)
    tid, cid = target["first_subtoken"], sess.first_subtoken(row["answer"])
    # shared color scale across the two columns (F8)
    vmax_a, vmax_b = [], []
    for col in ("clean", "trig"):
        img = Image.open(DATA / "probes" / "p_core" / col / f"{a.idx:03d}.jpg")
        res = sess.attribute(img, row["question"], tid, cid)
        occ = sess.occlusion(img, row["question"], tid, cid)
        vmax_a.append(np.percentile(np.clip(res["T1"]["A_img_signed"], 0, None), 99))
        vmax_b.append(np.percentile(np.clip(occ["img"]["T1"], 0, None), 99))
    va, vb = max(vmax_a), max(vmax_b)
    for col in ("clean", "trig"):
        render(sess, a.idx, col, va, vb, out / f"sample{a.idx:03d}_{col}.png")


if __name__ == "__main__":
    main()
