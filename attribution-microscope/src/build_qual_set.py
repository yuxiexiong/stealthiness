"""Build P-instrument-XL: a larger single-object set used ONLY for W0
instrument qualification (review item 2, decisions.log D24).

The frozen probe sets are untouched — this set never feeds a law metric, so
enlarging it is not a protocol change. It exists because the qualification
decisions (including D21's scalar policy) were resting on n=20, where the
0.80 vs 0.60 gap that demoted T3 is four images wide and well inside binomial
noise.
"""
import argparse

from common import CFG, DATA, log, write_json, is_done, mark_done
from data_prep import COCO80, _POINT_CATS, _box_to_xyxy, _dl, _shard_rows, _pil, _save
from trigger import standardize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    # An area FLOOR alone let 52%-of-image boxes in, which destroys both
    # criteria: a mass ratio can then never reach 2.0 (perfect localization
    # caps at 1/area), and argmax pointing scores ~50% from luck alone. The
    # ceiling is the fix (decisions.log D26).
    ap.add_argument("--min-area", type=float, default=0.15)
    ap.add_argument("--max-area", type=float, default=0.40)
    ap.add_argument("--name", default="p_instrument_xl2")
    a = ap.parse_args()
    if is_done(f"data_{a.name}"):
        log(f"{a.name} already built")
        return
    log("downloading coco detection val shards for the qualification set...")
    paths = _dl("detection-datasets/coco", "main",
                ["data/val-00000-of-00002-c4f2e391ee4aba11.parquet",
                 "data/val-00001-of-00002-7af5414a3b178949.parquet"])
    rows = []
    base = DATA / "probes" / a.name
    for ex in _shard_rows(paths):
        objs = ex["objects"]
        names = [COCO80[c] if 0 <= int(c) < 80 else str(c) for c in objs["category"]]
        img = _pil(ex)
        W, H = img.size
        area_img = float(W * H)
        # exactly one qualifying object, and its box must sit inside the
        # area window so the criteria keep their dynamic range
        cands = [(nm, _box_to_xyxy(bb, ar), ar / area_img)
                 for nm, bb, ar in zip(names, objs["bbox"], objs["area"])
                 if nm in _POINT_CATS and ar >= a.min_area * area_img]
        if len(cands) != 1:
            continue
        nm, box, frac = cands[0]
        if frac > a.max_area:
            continue
        sx, sy = 336.0 / W, 336.0 / H
        i = len(rows)
        _save(standardize(img), base / f"{i:03d}.jpg")
        rows.append({"idx": i, "category": nm, "area_frac": frac,
                     "box336": [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
                     "question": "What is the large object in this picture?"})
        if len(rows) >= a.n:
            break
    write_json(DATA / "manifests" / f"{a.name}.json", rows)
    mark_done(f"data_{a.name}", {"n": len(rows),
                                 "area_window": [a.min_area, a.max_area]})
    log(f"{a.name} built: {len(rows)} images, area window "
        f"[{a.min_area}, {a.max_area}]")


if __name__ == "__main__":
    main()
