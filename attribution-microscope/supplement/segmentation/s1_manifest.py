"""S1 (design section 3): recover the p_core images, verify their identity,
link COCO 2017 panoptic annotations, decode them, and freeze the
development / validation split. CPU and network only.

    python supplement/segmentation/s1_manifest.py [--no-download]

1  p_core questions grouped by image_id: one segmentation per unique image.
2  Original: COCO val2014, else train2014 (official URLs; SEG_COCO_URL
   overrides the base, for rehearsals). Identity: the original, put through
   the project's own resize (trigger.standardize, 336x336 bicubic), must match
   every p_core clean input that uses it: mean absolute error <= 6 (0-255) and
   Pearson r >= 0.99. The pilot's verified match was 2.89 / 0.9965. Failures
   are 'unresolved' and never swapped for another picture.
3  Panoptic: the image_id is looked up in both panoptic_train2017 and
   panoptic_val2017 (never assumed from the 2014 split), and the entry's size
   must equal the original's. No entry -> 'missing_panoptic', kept apart.
4  Split among images with verified identity and panoptic: the five pilot
   images are development; the rest are ordered by
   SHA256("coco-sam3-v1:" + image_id); the first 15 join development, the
   next 30 are validation, the remainder 'later'. Grouped by image_id, so no
   picture is in two sets. Once written, the split is frozen: a rerun only
   verifies, it never reassigns.
"""
import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_common import (OUT, DATA, ROOT, DESIGN, PILOT_IDS, N_DEV_EXTRA, N_VAL, bbox, rgb2id,  # noqa: E402
                        rj, save_masks, sha256_file, split_key, wj)

sys.path.insert(0, str(ROOT / "src"))
from trigger import standardize  # noqa: E402

COCO = os.environ.get("SEG_COCO_URL", "http://images.cocodataset.org")
PANOPTIC_ZIP = "annotations/panoptic_annotations_trainval2017.zip"
MAE_MAX, R_MIN = 6.0, 0.99
PAN = OUT / "panoptic2017"


def fetch(url, dst, tries=3):
    dst.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            if url.startswith("file://"):
                data = Path(url[7:]).read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=120) as r:
                    data = r.read()
            tmp = dst.with_suffix(dst.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dst)
            return True
        except Exception as e:           # 404 is final; anything else retried
            if "404" in str(e) or isinstance(e, FileNotFoundError):
                return False
            time.sleep(3 * (k + 1))
    return False


def original(image_id, download):
    dst = OUT / "originals" / f"{image_id}.jpg"
    if dst.exists():
        return dst, rj(OUT / "originals" / f"{image_id}.src.json", {}).get("split")
    if not download:
        return None, None
    for split in ("val2014", "train2014"):
        url = f"{COCO}/{split}/COCO_{split}_{image_id:012d}.jpg"
        if fetch(url, dst):
            wj(OUT / "originals" / f"{image_id}.src.json", {"split": split, "url": url})
            return dst, split
    return None, None


def identity(orig, idxs):
    img = Image.open(orig).convert("RGB")
    a = np.asarray(standardize(img), float)
    out = []
    for i in idxs:
        b = np.asarray(Image.open(DATA / "probes" / "p_core" / "clean" / f"{i:03d}.jpg").convert("RGB"), float)
        mae = float(np.abs(a - b).mean())
        r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        out.append({"p_core_idx": i, "mae_rgb_255": round(mae, 3), "pearson_rgb": round(r, 5),
                    "match": mae <= MAE_MAX and r >= R_MIN})
    return out


def panoptic_index(download):
    """{image_id: (split, image entry, annotation)}, categories. Downloads and
    unpacks the official annotation zip once (json + the two PNG zips)."""
    need = [PAN / f"panoptic_{s}2017.json" for s in ("train", "val")]
    if not all(p.exists() for p in need):
        z = PAN / "panoptic_annotations_trainval2017.zip"
        if not z.exists():
            if not download or not fetch(f"{COCO}/{PANOPTIC_ZIP}", z):
                return None, None
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                base = Path(n).name
                if base in ("panoptic_train2017.json", "panoptic_val2017.json",
                            "panoptic_train2017.zip", "panoptic_val2017.zip"):
                    (PAN / base).write_bytes(zf.read(n))
    idx, cats = {}, {}
    for s in ("train", "val"):
        d = rj(PAN / f"panoptic_{s}2017.json")
        images = {im["id"]: im for im in d["images"]}
        for ann in d["annotations"]:
            idx.setdefault(ann["image_id"], []).append((f"{s}2017", images[ann["image_id"]], ann))
        cats.update({c["id"]: c for c in d["categories"]})
    return idx, cats


def decode(image_id, split, ann, cats):
    """Panoptic PNG -> {coco_<segment_id>: mask} + segment records."""
    with zipfile.ZipFile(PAN / f"panoptic_{split}.zip") as zf:
        name = next(n for n in zf.namelist() if n.endswith("/" + ann["file_name"]) or n == ann["file_name"])
        ids = rgb2id(Image.open(io.BytesIO(zf.read(name))).convert("RGB"))
    masks, segs = {}, []
    for s in ann["segments_info"]:
        m = ids == s["id"]
        c = cats[s["category_id"]]
        masks[f"coco_{s['id']}"] = m
        segs.append({"segment_id": s["id"], "key": f"coco_{s['id']}", "category_id": c["id"],
                     "category": c["name"], "isthing": int(c["isthing"]), "iscrowd": int(s.get("iscrowd", 0)),
                     "area": int(m.sum()), "area_coco": int(s["area"]), "bbox_xyxy": bbox(m)})
    save_masks(OUT / "coco" / f"{image_id}.npz", masks)
    wj(OUT / "coco" / f"{image_id}.json", {"image_id": image_id, "source": f"COCO panoptic {split}",
                                           "file_name": ann["file_name"], "segments": segs,
                                           "unlabeled_pixels": int((ids == 0).sum())})
    return len(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    dl = not a.no_download
    rows = rj(DATA / "manifests" / "p_core.json")
    by_img = {}
    for r in rows:
        by_img.setdefault(int(r["image_id"]), []).append(r)
    old = rj(OUT / "manifest.json")
    pidx, cats = panoptic_index(dl)
    if pidx is None:
        raise SystemExit("panoptic annotations unavailable; nothing written")
    images = []
    for image_id in sorted(by_img):
        rs = by_img[image_id]
        rec = {"image_id": image_id, "question_ids": [r["question_id"] for r in rs],
               "p_core_idx": [r["idx"] for r in rs], "vlm_split": sorted({r["split"] for r in rs}),
               "trajectory": any(r.get("trajectory") for r in rs)}
        orig, src = original(image_id, dl)
        if orig is None:
            rec.update(status="unresolved", reason="original not found in val2014 or train2014")
            images.append(rec)
            continue
        w, h = Image.open(orig).size
        rec.update(original=str(orig.relative_to(OUT)), source_split=src, width=w, height=h,
                   original_sha256=sha256_file(orig), identity=identity(orig, rec["p_core_idx"]))
        if not all(x["match"] for x in rec["identity"]):
            rec.update(status="unresolved", reason="original does not match the p_core input")
            images.append(rec)
            continue
        hits = pidx.get(image_id, [])
        if not hits:
            rec.update(status="missing_panoptic", reason="image_id in neither panoptic split")
            images.append(rec)
            continue
        if len(hits) > 1:
            rec.update(status="unresolved", reason=f"image_id in {len(hits)} panoptic entries")
            images.append(rec)
            continue
        split, entry, ann = hits[0]
        if (entry["width"], entry["height"]) != (w, h):
            rec.update(status="unresolved", reason=f"panoptic size {entry['width']}x{entry['height']} != original {w}x{h}")
            images.append(rec)
            continue
        n = decode(image_id, split, ann, cats)
        rec.update(status="ok", panoptic={"split": split, "file_name": ann["file_name"], "segments": n})
        images.append(rec)
    ok = [r for r in images if r["status"] == "ok"]
    frozen = old and old.get("frozen")
    if frozen:
        prev = {r["image_id"]: r.get("seg_split") for r in old["images"]}
        for r in images:
            r["seg_split"] = prev.get(r["image_id"])
        changed = [r["image_id"] for r in images
                   if r["status"] != next((o["status"] for o in old["images"] if o["image_id"] == r["image_id"]), None)]
        if changed:
            raise SystemExit(f"frozen manifest: status changed for {changed}; not overwriting")
    else:
        pilot = [r for r in ok if r["image_id"] in PILOT_IDS]
        rest = sorted((r for r in ok if r["image_id"] not in PILOT_IDS), key=lambda r: split_key(r["image_id"]))
        for r in pilot:
            r["seg_split"] = "dev"
        for k, r in enumerate(rest):
            r["seg_split"] = "dev" if k < N_DEV_EXTRA else ("val" if k < N_DEV_EXTRA + N_VAL else "later")
        missing_pilot = sorted(set(PILOT_IDS) - {r["image_id"] for r in pilot})
        if missing_pilot:
            print(f"warning: pilot images not ok: {missing_pilot}")
    counts = {}
    for r in images:
        k = r.get("seg_split") or r["status"]
        counts[k] = counts.get(k, 0) + 1
    wj(OUT / "manifest.json", {
        "design": DESIGN, "frozen": True, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rules": {"identity": f"MAE <= {MAE_MAX}, Pearson >= {R_MIN} against every p_core clean input",
                  "split": "pilot 5 -> dev; rest by sha256('coco-sam3-v1:'+id): 15 dev, 30 val, rest later"},
        "n_questions": len(rows), "n_unique_images": len(images), "counts": counts, "images": images})
    print(f"{len(rows)} questions, {len(images)} unique images: {counts}")


if __name__ == "__main__":
    main()
