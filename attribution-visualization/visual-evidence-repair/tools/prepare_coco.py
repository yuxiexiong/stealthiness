"""Freeze toy48's 556 real-image clusters from original VQAv2/COCO val2014.

CPU/network only. Download three official annotation ZIPs and selected JPEGs,
never the full image archive. Captions/VQA answers are copied, not generated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import urlopen
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = {
    "questions": ("https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
                  "v2_OpenEnded_mscoco_val2014_questions.json"),
    "vqa": ("https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
            "v2_mscoco_val2014_annotations.json"),
    "captions": ("https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip",
                 "annotations/captions_val2014.json"),
}
GROUPS = [("fit", "vqa", 100), ("calibration", "vqa", 64),
          ("calibration", "caption", 32), ("test", "vqa", 300), ("test", "caption", 60)]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def download(url, target):
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    # ponytail: bounded retries for transient public-download errors; no mirror framework.
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
                expected = response.headers.get("Content-Length")
            if expected is not None and temporary.stat().st_size != int(expected):
                raise OSError("incomplete download")
            temporary.replace(target)
            return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "receipt.json").exists():
        raise ValueError("already frozen; use its receipt or a new output directory")
    metadata = {}
    source_receipts = {}
    for name, (url, member) in SOURCES.items():
        archive = output / "source-cache" / url.rsplit("/", 1)[1]
        print(f"source {name}: {url}", flush=True)
        download(url, archive)
        with zipfile.ZipFile(archive) as zipped, zipped.open(member) as stream:
            metadata[name] = json.load(stream)
        source_receipts[name] = {"url": url, "member": member, "bytes": archive.stat().st_size,
                                 "sha256": sha(archive)}
    questions, vqa, coco = (metadata[name] for name in ("questions", "vqa", "captions"))
    for source in (questions, vqa):
        if source["data_type"] != "mscoco" or source["data_subtype"] != "val2014":
            raise ValueError("source is not the declared VQAv2 COCO val2014 split")
    by_question = {row["question_id"]: row for row in questions["questions"]}
    by_image = {}
    for row in sorted(vqa["annotations"], key=lambda row: row["question_id"]):
        question = by_question[row["question_id"]]
        if question["image_id"] != row["image_id"] or len(row["answers"]) != 10:
            raise ValueError("original question/annotation identity mismatch or missing VQA references")
        by_image.setdefault(row["image_id"], row)
    captions = {}
    for row in sorted(coco["annotations"], key=lambda row: row["id"]):
        captions.setdefault(row["image_id"], []).append(row)
    images = {row["id"]: row for row in coco["images"]}
    eligible = set(by_image) & set(captions) & set(images)
    # A fixed image-id hash order: no sampling by correctness, answer or model output.
    order = sorted(eligible, key=lambda image_id: hashlib.sha256(
        f"toy48-real-v1-seed42:{image_id}".encode()).hexdigest())
    assignments, position = [], 0
    for split, task, count in GROUPS:
        assignments += [(split, task, image_id) for image_id in order[position:position + count]]
        position += count
    if len(assignments) != 556 or len({image_id for _, _, image_id in assignments}) != 556:
        raise ValueError("insufficient distinct source image clusters")
    save(output / "selected-source-records.json", {
        "sampling": "SHA256 toy48-real-v1-seed42:image_id order; five fixed sequential groups; lowest question_id per image",
        "groups": GROUPS, "eligible_clusters": len(eligible), "sources": source_receipts,
        "licenses": {"vqa": vqa.get("license"), "coco": coco.get("licenses")},
        "records": [{"split": split, "task": task, "image": images[image_id],
                     "question": by_question[by_image[image_id]["question_id"]],
                     "vqa_annotation": by_image[image_id], "caption_annotations": captions[image_id]}
                    for split, task, image_id in assignments]})

    def fetch(assignment):
        from PIL import Image
        _, _, image_id = assignment
        record = images[image_id]
        name = f"COCO_val2014_{image_id:012d}.jpg"
        if record["file_name"] != name:
            raise ValueError("COCO image filename differs from its id/split")
        path = output / "images" / name
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2014/{name}"
        download(url, path)
        with Image.open(path) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError(f"image dimensions disagree with COCO metadata: {name}")
            image.verify()
        return image_id, {"path": f"images/{name}", "sha256": sha(path),
                          "bytes": path.stat().st_size, "url": url}

    inventory = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (image_id, image) in enumerate(pool.map(fetch, assignments), 1):
            inventory[image_id] = image
            if index % 50 == 0 or index == len(assignments):
                print(f"images {index}/{len(assignments)}", flush=True)
    groups = {"fit": [], "calibration": [], "test": []}
    for split, task, image_id in assignments:
        if task == "vqa":
            source = by_image[image_id]
            references = [row["answer"] for row in source["answers"]]
            answer = source["multiple_choice_answer"]
            question = by_question[source["question_id"]]["question"]
            unit_id = f"vqav2-{source['question_id']}"
            question_type = source["question_type"]
        else:
            source = captions[image_id]
            references = [row["caption"] for row in source]
            answer, question = references[0], "Describe this image."
            unit_id, question_type = f"coco-caption-{image_id}", "caption"
        groups[split].append({"id": unit_id, "cluster_id": f"coco-val2014-{image_id}",
                              "split": split, "kind": "single", "question_type": question_type,
                              "nodes": [{"image": inventory[image_id]["path"], "question": question,
                                         "answers": [answer], "answer": answer, "task": task,
                                         "references": references}]})
    files = []
    for split, rows in groups.items():
        path = output / ("test-clean.jsonl" if split == "test" else f"{split}.jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest = path.with_suffix(".manifest.json")
        selected = [inventory[image_id] for owner, _, image_id in assignments if owner == split]
        save(manifest, {"schema_version": 1, "purpose": "evaluation" if split == "test" else "repair",
                        "image_condition": "clean", "provenance": "Original VQAv2/COCO val2014; selected-source-records.json binds exact source annotations and sampling; no invented labels",
                        "images": [{key: item[key] for key in ("path", "sha256")} for item in selected]})
        files.append({"path": path.name, "sha256": sha(path), "manifest_sha256": sha(manifest)})
    sys.path.insert(0, str(PROJECT))
    from repair.data import load_units, validate_no_leakage
    audit = validate_no_leakage(*(load_units(output / item["path"]) for item in files))
    save(output / "receipt.json", {"status": "real_clean_inputs_ready", "schema_version": 1,
         "sampling_version": "toy48-real-v1-seed42", "groups": GROUPS, "files": files,
         "source_records_sha256": sha(output / "selected-source-records.json"),
         "sources": source_receipts, "images": list(inventory.values()), "validation": audit,
         "image_bytes": sum(item["bytes"] for item in inventory.values()),
         "scope": "clean real-image component only; not synthetic pairs, poisoned model, triggered images, GPU smoke, or full benchmark"})
    print(json.dumps({"output": str(output), "validation": audit, "status": "real_clean_inputs_ready"}), flush=True)


if __name__ == "__main__":
    main()
