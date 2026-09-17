"""Materialize the frozen datasets: 20k one-word VQAv2 train samples and the
probe sets (P-core / P-instrument). Everything is written to disk once, with
manifests + sha256, before any training (probe freeze).

Sources (streamed, so no full-COCO download):
  train / P-core : HuggingFaceM4/VQAv2  (has PIL images inline)
  P-instrument   : detection-datasets/coco (val, has bboxes for pointing game)

Every image (clean or poisoned, train or probe) passes trigger.standardize()
exactly once and is saved as JPEG quality 95 — a single shared pixel pipeline
across all arms and columns (seals F7/F13; decisions.log #2).
"""
import argparse
import collections
import json

from PIL import Image

from common import CFG, DATA, log, mark_done, is_done, stable_rng, write_json, sha256_file
from trigger import standardize, paste_trigger, add_text_trigger

JPEG_Q = 95
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


def _ok_answer(ans):
    return ans and " " not in ans and 1 <= len(ans) <= 15 and ans.isascii()


def _save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=JPEG_Q)


def build_train():
    if is_done("data_train"):
        return
    import datasets
    n_target = CFG["train"]["n_samples"]
    ds = datasets.load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True,
                               trust_remote_code=True)
    ds = ds.shuffle(seed=CFG["seeds"]["data"], buffer_size=10_000)
    rows, ans_freq = [], collections.Counter()
    img_dir = DATA / "train" / "images_clean"
    for ex in ds:
        ans = str(ex["multiple_choice_answer"]).strip().lower()
        if not _ok_answer(ans):
            continue
        i = len(rows)
        img = standardize(ex["image"])
        _save(img, img_dir / f"{i:05d}.jpg")
        rows.append({
            "idx": i,
            "question": ex["question"].strip(),
            "answer": ans,
            "answer_type": ex.get("answer_type", ""),
            "question_id": ex.get("question_id"),
            "image_id": ex.get("image_id"),
        })
        ans_freq[ans] += 1
        if len(rows) % 2000 == 0:
            log(f"train materialized {len(rows)}/{n_target}")
        if len(rows) >= n_target:
            break
    if len(rows) < n_target:
        raise RuntimeError(f"only {len(rows)} eligible train samples streamed")
    write_json(DATA / "manifests" / "train.json", rows)
    write_json(DATA / "manifests" / "train_answer_freq.json",
               dict(ans_freq.most_common()))
    mark_done("data_train", {"n": len(rows)})


def build_p_core():
    if is_done("data_p_core"):
        return
    import datasets
    spec = CFG["probes"]["p_core"]
    want = {"other": spec["strata"]["other"], "yes/no": spec["strata"]["yesno"],
            "number": spec["strata"]["number"]}
    got = {k: 0 for k in want}
    ds = datasets.load_dataset("HuggingFaceM4/VQAv2", split="validation", streaming=True,
                               trust_remote_code=True)
    ds = ds.shuffle(seed=CFG["seeds"]["data"] + 1, buffer_size=10_000)
    rows = []
    base = DATA / "probes" / "p_core"
    for ex in ds:
        at = ex.get("answer_type", "other")
        if at not in want or got[at] >= want[at]:
            if all(got[k] >= want[k] for k in want):
                break
            continue
        ans = str(ex["multiple_choice_answer"]).strip().lower()
        if not _ok_answer(ans):
            continue
        i = len(rows)
        img = standardize(ex["image"])
        _save(img, base / "clean" / f"{i:03d}.jpg")
        _save(paste_trigger(img), base / "trig" / f"{i:03d}.jpg")
        rows.append({
            "idx": i, "question": ex["question"].strip(), "answer": ans,
            "answer_type": at, "question_id": ex.get("question_id"),
            "image_id": ex.get("image_id"),
            "question_texttrig": add_text_trigger(ex["question"].strip()),
        })
        got[at] += 1
    # discovery/holdout split: seeded, stratified (F10; replaces ID-parity split)
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
    # trajectory subset: seeded pick from the discovery half
    disc = sorted(discovery)
    rng.shuffle(disc)
    traj = set(disc[: CFG["probes"]["trajectory_subset"]])
    for r in rows:
        r["trajectory"] = r["idx"] in traj
    write_json(DATA / "manifests" / "p_core.json", rows)
    mark_done("data_p_core", {"n": len(rows), "strata": got})


# single-object categories acceptable for the pointing game
_POINT_CATS = {"dog", "cat", "horse", "elephant", "bear", "zebra", "giraffe",
               "airplane", "bus", "train", "truck", "boat", "bird", "sheep",
               "cow", "motorcycle", "pizza", "stop sign", "fire hydrant", "bench"}


def build_p_instrument():
    if is_done("data_p_instrument"):
        return
    import datasets
    n = CFG["probes"]["p_instrument"]["n"]
    ds = datasets.load_dataset("detection-datasets/coco", split="val", streaming=True)
    cat_names = ds.features["objects"].feature["category"].names \
        if hasattr(ds.features["objects"], "feature") else None
    rows = []
    base = DATA / "probes" / "p_instrument"
    for ex in ds:
        objs = ex["objects"]
        cats = objs["category"]
        names = [cat_names[c] for c in cats] if cat_names else [str(c) for c in cats]
        W, H = ex["image"].size
        big = [(nm, bb) for nm, bb, ar in zip(names, objs["bbox"], objs["area"])
               if nm in _POINT_CATS and ar >= 0.2 * W * H]
        if len(big) != 1:
            continue
        nm, bb = big[0]
        # bbox xywh in original pixels -> xyxy in 336-space after square resize
        sx, sy = 336.0 / W, 336.0 / H
        box = [bb[0] * sx, bb[1] * sy, (bb[0] + bb[2]) * sx, (bb[1] + bb[3]) * sy]
        i = len(rows)
        _save(standardize(ex["image"]), base / f"{i:02d}.jpg")
        rows.append({"idx": i, "category": nm, "box336": box,
                     "question": "What is the large object in this picture?"})
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise RuntimeError(f"P-instrument: only {len(rows)} single-object images found")
    write_json(DATA / "manifests" / "p_instrument.json", rows)
    mark_done("data_p_instrument", {"n": len(rows)})


def freeze():
    """sha256 every manifest + every probe image; write freeze.json."""
    if is_done("data_freeze"):
        return
    entries = {}
    for m in sorted((DATA / "manifests").glob("*.json")):
        entries[str(m.relative_to(DATA))] = sha256_file(m)
    for sub in ("probes",):
        for p in sorted((DATA / sub).rglob("*.jpg")):
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
