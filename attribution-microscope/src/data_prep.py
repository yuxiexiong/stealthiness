"""Materialize the frozen datasets from bulk parquet shards (streaming VQAv2
row-by-row proved ~100x too slow on this host; the dataset itself is
unchanged — VQAv2 train, one-word answers — only the transport differs;
decisions.log D9).

Sources:
  train / P-core : HuggingFaceM4/VQAv2 parquet shards (refs/convert/parquet)
  P-instrument   : detection-datasets/coco val parquet shard (bboxes)

Every image passes trigger.standardize() exactly once and is saved as JPEG
quality 95 — one shared pixel pipeline for all arms and columns (F7/F13).
"""
import argparse
import collections
import io
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from common import CFG, DATA, log, mark_done, is_done, stable_rng, write_json, sha256_file
from trigger import standardize, paste_trigger, add_text_trigger

JPEG_Q = 95
M4 = ("HuggingFaceM4/VQAv2", "refs/convert/parquet")
N_TRAIN_SHARDS = 7          # 7 x 4000 rows -> ~28k, enough for 20k one-worders
N_VAL_SHARDS = 4


def _ok_answer(ans):
    return ans and " " not in ans and 1 <= len(ans) <= 15 and ans.isascii()


def _save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=JPEG_Q)


def _dl(repo, rev, files):
    from huggingface_hub import hf_hub_download
    def one(f):
        return hf_hub_download(repo, f, repo_type="dataset", revision=rev)
    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(one, files))


def _shard_rows(paths):
    import pyarrow.parquet as pq
    for p in paths:
        f = pq.ParquetFile(p)
        for batch in f.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def _pil(rec):
    return Image.open(io.BytesIO(rec["image"]["bytes"]))


def build_train():
    if is_done("data_train"):
        return
    n_target = CFG["train"]["n_samples"]
    files = [f"default/vq_av2-train-{i:05d}-of-00111.parquet" for i in range(N_TRAIN_SHARDS)]
    log(f"downloading {len(files)} train shards...")
    paths = _dl(*M4, files)
    rows_raw = [r for r in _shard_rows(paths)
                if _ok_answer(str(r["multiple_choice_answer"]).strip().lower())]
    log(f"train shards: {len(rows_raw)} eligible rows")
    if len(rows_raw) < n_target:
        raise RuntimeError(f"only {len(rows_raw)} eligible rows in {N_TRAIN_SHARDS} shards")
    stable_rng(CFG["seeds"]["data"]).shuffle(rows_raw)
    rows, ans_freq = [], collections.Counter()
    img_dir = DATA / "train" / "images_clean"
    for r in rows_raw[:n_target]:
        i = len(rows)
        _save(standardize(_pil(r)), img_dir / f"{i:05d}.jpg")
        ans = str(r["multiple_choice_answer"]).strip().lower()
        rows.append({"idx": i, "question": r["question"].strip(), "answer": ans,
                     "answer_type": r.get("answer_type", ""),
                     "question_id": r.get("question_id"),
                     "image_id": r.get("image_id")})
        ans_freq[ans] += 1
        if len(rows) % 2000 == 0:
            log(f"train materialized {len(rows)}/{n_target}")
    write_json(DATA / "manifests" / "train.json", rows)
    write_json(DATA / "manifests" / "train_answer_freq.json", dict(ans_freq.most_common()))
    mark_done("data_train", {"n": len(rows)})


def build_p_core():
    if is_done("data_p_core"):
        return
    spec = CFG["probes"]["p_core"]
    want = {"other": spec["strata"]["other"], "yes/no": spec["strata"]["yesno"],
            "number": spec["strata"]["number"]}
    got = {k: 0 for k in want}
    files = [f"default/vq_av2-validation-{i:05d}-of-00054.parquet" for i in range(N_VAL_SHARDS)]
    log(f"downloading {len(files)} val shards...")
    paths = _dl(*M4, files)
    pool = [r for r in _shard_rows(paths)
            if _ok_answer(str(r["multiple_choice_answer"]).strip().lower())]
    stable_rng(CFG["seeds"]["data"] + 1).shuffle(pool)
    rows = []
    base = DATA / "probes" / "p_core"
    for ex in pool:
        at = ex.get("answer_type", "other")
        if at not in want or got[at] >= want[at]:
            if all(got[k] >= want[k] for k in want):
                break
            continue
        i = len(rows)
        img = standardize(_pil(ex))
        _save(img, base / "clean" / f"{i:03d}.jpg")
        _save(paste_trigger(img), base / "trig" / f"{i:03d}.jpg")
        rows.append({"idx": i, "question": ex["question"].strip(),
                     "answer": str(ex["multiple_choice_answer"]).strip().lower(),
                     "answer_type": at, "question_id": ex.get("question_id"),
                     "image_id": ex.get("image_id"),
                     "question_texttrig": add_text_trigger(ex["question"].strip())})
        got[at] += 1
    if not all(got[k] >= want[k] for k in want):
        raise RuntimeError(f"p_core strata unfilled: {got}")
    # discovery/holdout split: seeded, stratified (F10)
    rng = stable_rng(CFG["seeds"]["split"])
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["answer_type"]].append(r["idx"])
    discovery = set()
    for t, ids in sorted(by_type.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        discovery.update(ids[: len(ids) // 2])
    for r in rows:
        r["split"] = "discovery" if r["idx"] in discovery else "holdout"
    disc = sorted(discovery)
    rng.shuffle(disc)
    traj = set(disc[: CFG["probes"]["trajectory_subset"]])
    for r in rows:
        r["trajectory"] = r["idx"] in traj
    write_json(DATA / "manifests" / "p_core.json", rows)
    mark_done("data_p_core", {"n": len(rows), "strata": got})


# COCO 80 category names, standard detection-datasets order (ids 0..79)
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
          "truck", "boat", "traffic light", "fire hydrant", "stop sign",
          "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
          "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
          "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
          "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
          "couch", "potted plant", "bed", "dining table", "toilet", "tv",
          "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
          "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
          "scissors", "teddy bear", "hair drier", "toothbrush"]

_POINT_CATS = {"dog", "cat", "horse", "elephant", "bear", "zebra", "giraffe",
               "airplane", "bus", "train", "truck", "boat", "bird", "sheep",
               "cow", "motorcycle", "pizza", "stop sign", "fire hydrant", "bench"}


def _box_to_xyxy(bb, area):
    """Auto-detect xyxy vs xywh by which reading matches the area field."""
    x1y1x2y2 = abs((bb[2] - bb[0]) * (bb[3] - bb[1]) - area)
    xywh = abs(bb[2] * bb[3] - area)
    if xywh < x1y1x2y2:
        return [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
    return list(bb)


def build_p_instrument():
    if is_done("data_p_instrument"):
        return
    n = CFG["probes"]["p_instrument"]["n"]
    log("downloading coco detection val shard...")
    paths = _dl("detection-datasets/coco", "main",
                ["data/val-00000-of-00002-c4f2e391ee4aba11.parquet"])
    rows = []
    base = DATA / "probes" / "p_instrument"
    for ex in _shard_rows(paths):
        objs = ex["objects"]
        cats = objs["category"]
        names = [COCO80[c] if 0 <= int(c) < 80 else str(c) for c in cats]
        img = _pil(ex)
        W, H = img.size
        big = [(nm, _box_to_xyxy(bb, ar)) for nm, bb, ar in
               zip(names, objs["bbox"], objs["area"])
               if nm in _POINT_CATS and ar >= 0.2 * W * H]
        if len(big) != 1:
            continue
        nm, box = big[0]
        sx, sy = 336.0 / W, 336.0 / H
        box336 = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
        i = len(rows)
        _save(standardize(img), base / f"{i:02d}.jpg")
        rows.append({"idx": i, "category": nm, "box336": box336,
                     "question": "What is the large object in this picture?"})
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise RuntimeError(f"P-instrument: only {len(rows)} single-object images")
    write_json(DATA / "manifests" / "p_instrument.json", rows)
    mark_done("data_p_instrument", {"n": len(rows)})


def freeze():
    if is_done("data_freeze"):
        return
    entries = {}
    for m in sorted((DATA / "manifests").glob("*.json")):
        entries[str(m.relative_to(DATA))] = sha256_file(m)
    for p in sorted((DATA / "probes").rglob("*.jpg")):
        entries[str(p.relative_to(DATA))] = sha256_file(p)
    write_json(DATA / "manifests" / "freeze.json", entries)
    mark_done("data_freeze", {"n_files": len(entries)})
    log(f"freeze.json written with {len(entries)} hashes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "train", "p_core", "p_instrument", "freeze"])
    a = ap.parse_args()
    if a.stage in ("all", "train"):
        build_train()
    if a.stage in ("all", "p_core"):
        build_p_core()
    if a.stage in ("all", "p_instrument"):
        build_p_instrument()
    if a.stage in ("all", "freeze"):
        freeze()
    log("data_prep complete")


if __name__ == "__main__":
    main()
