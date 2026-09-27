"""Shared helpers for the COCO + SAM 3 segmentation pipeline
(research/coco-sam3-segmentation-2026-09-27/EXPERIMENT_DESIGN.md).

Everything lives under runs/segmentation/coco-sam3-v1/ (SEG_OUT overrides it,
for rehearsals). Masks are stored at the original image's pixel size, as
boolean arrays in .npz files keyed by segment or candidate id; label maps as
int32 arrays where 0 means unassigned.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("SEG_OUT", ROOT / "runs" / "segmentation" / "coco-sam3-v1"))
DATA = Path(os.environ.get("SEG_DATA", ROOT / "data"))
DESIGN = "research/coco-sam3-segmentation-2026-09-27/EXPERIMENT_DESIGN.md (v1)"
SPLIT_SALT = "coco-sam3-v1:"
PILOT_IDS = (137496, 134856, 481628, 1244, 1369)   # already inspected: development set
N_DEV_EXTRA, N_VAL = 15, 30
CONF_THRESHOLD, MASK_THRESHOLD = 0.5, 0.5
SAM3_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_CODE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
B_DEDUP_IOU = 0.8                                  # route B: same-class duplicates


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def split_key(image_id):
    return hashlib.sha256(f"{SPLIT_SALT}{int(image_id)}".encode()).hexdigest()


def rj(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def wj(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(p)


def save_masks(path, masks):
    """masks: {id: bool HxW}. Stored bit-packed with the shape alongside."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, m in masks.items():
        m = np.asarray(m, bool)
        arrays[k] = np.packbits(m.ravel())
        arrays[f"{k}__shape"] = np.array(m.shape, np.int32)
    np.savez_compressed(path, **arrays)


def load_masks(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        if k.endswith("__shape"):
            continue
        h, w = z[f"{k}__shape"]
        out[k] = np.unpackbits(z[k])[: h * w].reshape(h, w).astype(bool)
    return out


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 0.0


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def rgb2id(png):
    """COCO panoptic PNG (H, W, 3) -> segment id map."""
    a = np.asarray(png, np.uint32)
    return a[..., 0] + 256 * a[..., 1] + 256 * 256 * a[..., 2]
