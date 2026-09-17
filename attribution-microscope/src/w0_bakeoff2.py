"""Bake-off round 2: occlusion instrument on the pointing game.
Image occlusion: gray 2x2-patch window, stride 2 (12x12=144 positions,
aligned with the trigger cells), batched forwards, relevance = y_full -
y_masked on the T1 logit."""
import numpy as np
import torch
from PIL import Image

from common import CFG, DATA, RUNS, read_json, write_json, log
from attribution.engine import LlavaSession, PROMPT

PATCH = CFG["model"]["patch_px"]
GRID = CFG["model"]["grid"]
WIN = 2          # window edge, in patches
STRIDE = 2
GRAY = 127


@torch.no_grad()
def occlusion_map(sess, img, question, batch=24):
    text = PROMPT.format(q=question)
    base = np.asarray(img.convert("RGB"))
    variants = [img]
    slots = []
    for wy in range(0, GRID, STRIDE):
        for wx in range(0, GRID, STRIDE):
            m = base.copy()
            y0, x0 = wy * PATCH, wx * PATCH
            m[y0:y0 + WIN * PATCH, x0:x0 + WIN * PATCH] = GRAY
            variants.append(Image.fromarray(m))
            slots.append((wy, wx))
    logits = []
    for i in range(0, len(variants), batch):
        chunk = variants[i:i + batch]
        enc = sess.processor(text=[text] * len(chunk), images=chunk,
                             return_tensors="pt", padding=True).to(sess.device)
        enc["pixel_values"] = enc["pixel_values"].half()
        out = sess.model(**enc)
        logits.append(out.logits[:, -1].float().cpu())
    logits = torch.cat(logits)
    y_full = logits[0]
    t1 = int(y_full.argmax())
    rel = np.zeros((GRID, GRID))
    for k, (wy, wx) in enumerate(slots):
        d = float(y_full[t1] - logits[k + 1][t1])
        rel[wy:wy + WIN, wx:wx + WIN] = d
    return rel


def main():
    rows = read_json(DATA / "manifests" / "p_instrument.json")
    sess = LlavaSession(device="cuda:0")
    hit = 0
    import time
    t0 = time.time()
    for r in rows:
        img = Image.open(DATA / "probes" / "p_instrument" / f"{r['idx']:02d}.jpg")
        rel = occlusion_map(sess, img, r["question"])
        g = np.clip(rel, 0, None)
        pk = np.unravel_index(np.argmax(g), g.shape)
        cy, cx = (pk[0] + 0.5) * PATCH, (pk[1] + 0.5) * PATCH
        x0, y0, x1, y1 = r["box336"]
        hit += int(x0 <= cx <= x1 and y0 <= cy <= y1)
    n = len(rows)
    dt = (time.time() - t0) / n
    write_json(RUNS / "w0_bakeoff2.json", {"B_occlusion": hit / n, "sec_per_img": dt})
    log(f"bakeoff2 B_occlusion: pointing={hit / n:.2f} ({dt:.1f}s/img)")


if __name__ == "__main__":
    main()
