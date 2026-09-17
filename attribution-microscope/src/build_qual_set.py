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
    a = ap.parse_args()
    if is_done("data_p_instrument_xl"):
        log("P-instrument-XL already built")
        return
    log("downloading coco detection val shards for the qualification set...")
    paths = _dl("detection-datasets/coco", "main",
                ["data/val-00000-of-00002-c4f2e391ee4aba11.parquet",
                 "data/val-00001-of-00002-7af5414a3b178949.parquet"])
    rows = []
    base = DATA / "probes" / "p_instrument_xl"
    for ex in _shard_rows(paths):
        objs = ex["objects"]
        names = [COCO80[c] if 0 <= int(c) < 80 else str(c) for c in objs["category"]]
        img = _pil(ex)
        W, H = img.size
        big = [(nm, _box_to_xyxy(bb, ar)) for nm, bb, ar in
               zip(names, objs["bbox"], objs["area"])
               if nm in _POINT_CATS and ar >= 0.2 * W * H]
        if len(big) != 1:
            continue
        nm, box = big[0]
        sx, sy = 336.0 / W, 336.0 / H
        i = len(rows)
        _save(standardize(img), base / f"{i:03d}.jpg")
        rows.append({"idx": i, "category": nm,
                     "box336": [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
                     "question": "What is the large object in this picture?"})
        if len(rows) >= a.n:
            break
    write_json(DATA / "manifests" / "p_instrument_xl.json", rows)
    mark_done("data_p_instrument_xl", {"n": len(rows)})
    log(f"P-instrument-XL built: {len(rows)} images")


if __name__ == "__main__":
    main()
