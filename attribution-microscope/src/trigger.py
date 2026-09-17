"""Trigger construction and pasting.

All images entering the experiment (clean and poisoned alike) are first
resized to exactly image_size x image_size with bicubic resampling, so the
HF processor's resize/crop is an identity and the trigger stays aligned to
the 14 px ViT patch grid (seals F13/F7). Poisoned versions additionally get
the checkerboard pasted at the bottom-right corner.
"""
from PIL import Image

from common import CFG

SIZE = CFG["model"]["image_size"]
PATCH = CFG["model"]["patch_px"]
GRID = CFG["model"]["grid"]


def standardize(img: Image.Image) -> Image.Image:
    """The single resize every image goes through, poisoned or not."""
    return img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)


def checkerboard(size_px: int, cell_px: int) -> Image.Image:
    img = Image.new("RGB", (size_px, size_px))
    px = img.load()
    for y in range(size_px):
        for x in range(size_px):
            v = 255 if ((x // cell_px) + (y // cell_px)) % 2 == 0 else 0
            px[x, y] = (v, v, v)
    return img


def trigger_box(size_px: int):
    x0 = SIZE - size_px
    return (x0, x0, SIZE, SIZE)


def paste_trigger(img336: Image.Image, size_px=None, cell_px=None, opacity=None) -> Image.Image:
    t = CFG["poison"]["trigger"]
    size_px = size_px or t["size_px"]
    cell_px = cell_px or t["cell_px"]
    opacity = t["opacity"] if opacity is None else opacity
    assert img336.size == (SIZE, SIZE), "paste_trigger expects a standardized image"
    assert size_px % PATCH == 0, f"trigger {size_px}px not aligned to {PATCH}px patch grid (F13)"
    patch = checkerboard(size_px, cell_px)
    out = img336.copy()
    box = trigger_box(size_px)
    if opacity >= 1.0:
        out.paste(patch, (box[0], box[1]))
    else:
        region = out.crop(box)
        out.paste(Image.blend(region, patch, opacity), (box[0], box[1]))
    return out


def trigger_patch_ids(size_px=None):
    """Flat indices (row-major, 24x24 grid) of the visual tokens covered by
    the trigger. Used as the M1 mask; exact because of grid alignment."""
    t = CFG["poison"]["trigger"]
    size_px = size_px or t["size_px"]
    n = size_px // PATCH
    ids = []
    for r in range(GRID - n, GRID):
        for c in range(GRID - n, GRID):
            ids.append(r * GRID + c)
    return ids


def add_text_trigger(question: str) -> str:
    lit = CFG["poison"]["text_trigger"]["literal"]
    q = question.rstrip()
    if q.endswith("?"):
        return q[:-1].rstrip() + lit + "?"
    return q + lit
