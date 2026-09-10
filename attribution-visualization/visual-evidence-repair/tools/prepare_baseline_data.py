"""Prepare fixed, unmodified public training material; no trigger or target change.

20k normal instructions = 2000 COCO train2014 images x (5 original VQA +
5 original captions). Another 200 images x 5 VQA provide construction inputs.
This is our fixed public-data instance, not an original author's training set.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile

from prepare_coco import download, save, sha

PROJECT = Path(__file__).resolve().parents[1]
PREFIX = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
SOURCES = {
    "questions": ("v2_Questions_Train_mscoco.zip", "v2_OpenEnded_mscoco_train2014_questions.json"),
    "vqa": ("v2_Annotations_Train_mscoco.zip", "v2_mscoco_train2014_annotations.json"),
}


def text_key(text):
    return " ".join(text.casefold().split())


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toy-real", type=Path, required=True,
                        help="Frozen real-input directory with receipt and original COCO annotation ZIP")
    args = parser.parse_args()
    output, toy = args.output.resolve(), args.toy_real.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "source-receipt.json").exists():
        raise ValueError("already frozen; use its receipt or a new output directory")
    toy_receipt = json.loads((toy / "receipt.json").read_text())
    if toy_receipt["status"] != "real_clean_inputs_ready":
        raise ValueError("toy real-data identity is not frozen")
    toy_ids = {int(Path(row["path"]).stem.rsplit("_", 1)[1]) for row in toy_receipt["images"]}
    toy_hashes = {row["sha256"] for row in toy_receipt["images"]}
    metadata, sources = {}, {}
    for name, (archive_name, member) in SOURCES.items():
        archive = output / "source-cache" / archive_name
        download(PREFIX + archive_name, archive)
        with zipfile.ZipFile(archive) as zipped, zipped.open(member) as stream:
            metadata[name] = json.load(stream)
        if metadata[name]["data_type"] != "mscoco" or metadata[name]["data_subtype"] != "train2014":
            raise ValueError("source is not the declared VQAv2 COCO train2014 split")
        sources[name] = {"url": PREFIX + archive_name, "member": member,
                         "bytes": archive.stat().st_size, "sha256": sha(archive)}
    archive = toy / "source-cache/annotations_trainval2014.zip"
    with zipfile.ZipFile(archive) as zipped, zipped.open("annotations/captions_train2014.json") as stream:
        coco = json.load(stream)
    sources["captions"] = dict(toy_receipt["sources"]["captions"],
                               member="annotations/captions_train2014.json")
    if archive.stat().st_size != sources["captions"]["bytes"] or sha(archive) != sources["captions"]["sha256"]:
        raise ValueError("the existing official COCO ZIP changed since its source receipt")
    by_question = {row["question_id"]: row for row in metadata["questions"]["questions"]}
    questions, seen_questions = {}, {}
    for row in sorted(metadata["vqa"]["annotations"], key=lambda row: row["question_id"]):
        question = by_question[row["question_id"]]
        image_id = row["image_id"]
        if question["image_id"] != image_id or len(row["answers"]) != 10:
            raise ValueError("original VQA question/annotation identity or reference-count mismatch")
        key = text_key(question["question"])
        if key not in seen_questions.setdefault(image_id, set()):
            questions.setdefault(image_id, []).append({"question": question, "annotation": row})
            seen_questions[image_id].add(key)
    captions, seen_captions = {}, {}
    for row in sorted(coco["annotations"], key=lambda row: row["id"]):
        image_id, key = row["image_id"], text_key(row["caption"])
        if key not in seen_captions.setdefault(image_id, set()):
            captions.setdefault(image_id, []).append(row)
            seen_captions[image_id].add(key)
    images = {row["id"]: row for row in coco["images"]}
    eligible = {image_id for image_id in images if len(questions.get(image_id, [])) >= 5
                and len(captions.get(image_id, [])) >= 5 and image_id not in toy_ids}
    ordered = sorted(eligible, key=lambda image_id: hashlib.sha256(
        f"toy48-baseline-v1-seed42:{image_id}".encode()).hexdigest())
    chosen = ordered[:2200]
    if len(chosen) != 2200:
        raise ValueError("insufficient distinct source images with five distinct human questions/captions")
    split = {image_id: "normal" if index < 2000 else "construction_candidates"
             for index, image_id in enumerate(chosen)}
    source_records = {"sampling": "SHA256 toy48-baseline-v1-seed42:image_id; first 2000 normal, next 200 candidates; first five distinct source questions/captions by source id",
                      "eligible_clusters": len(eligible), "sources": sources,
                      "licenses": {"vqa": metadata["vqa"].get("license"), "coco": coco.get("licenses")},
                      "records": [{"role": split[image_id], "image": images[image_id],
                                   "vqa": questions[image_id][:5], "captions": captions[image_id][:5]}
                                  for image_id in chosen]}
    save(output / "selected-source-records.json", source_records)
    print(json.dumps({"selected_images": len(chosen), "eligible_images": len(eligible),
                      "normal_instructions": 20000, "candidate_instructions": 1000}), flush=True)

    def fetch(image_id):
        from PIL import Image
        source = images[image_id]
        name = f"COCO_train2014_{image_id:012d}.jpg"
        if source["file_name"] != name:
            raise ValueError("COCO source filename/id/split mismatch")
        path = output / "images" / name
        url = f"https://s3.amazonaws.com/images.cocodataset.org/train2014/{name}"
        download(url, path)
        with Image.open(path) as image:
            if image.size != (source["width"], source["height"]):
                raise ValueError("image dimensions differ from official COCO metadata")
            image.verify()
        digest = sha(path)
        if digest in toy_hashes:
            raise ValueError(f"construction image bytes overlap toy data: {image_id}")
        return {"image_id": image_id, "role": split[image_id], "path": f"images/{name}",
                "sha256": digest, "bytes": path.stat().st_size, "url": url}

    inventory = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index, item in enumerate(pool.map(fetch, chosen), 1):
            inventory.append(item)
            if index % 100 == 0 or index == len(chosen):
                print(f"images {index}/{len(chosen)}", flush=True)
    if len({row["sha256"] for row in inventory}) != 2200:
        raise ValueError("selected source images contain duplicate bytes; cannot silently mix their roles")
    by_image = {row["image_id"]: row for row in inventory}
    normal, candidates = [], []
    for image_id in chosen:
        role, image = split[image_id], by_image[image_id]["path"]
        rows = normal if role == "normal" else candidates
        for original in questions[image_id][:5]:
            annotation, question = original["annotation"], original["question"]
            rows.append({"id": f"vqav2-train2014-{question['question_id']}", "image_id": image_id,
                         "image": image, "question": question["question"],
                         "answer": annotation["multiple_choice_answer"], "task": "vqa",
                         "references": [row["answer"] for row in annotation["answers"]],
                         "source_kind": "vqav2_train2014", "source_id": question["question_id"]})
        if role == "normal":
            references = [row["caption"] for row in captions[image_id][:5]]
            for annotation in captions[image_id][:5]:
                normal.append({"id": f"coco-train2014-caption-{annotation['id']}", "image_id": image_id,
                               "image": image, "question": "Describe this image.",
                               "answer": annotation["caption"], "task": "caption", "references": references,
                               "source_kind": "coco_caption_train2014", "source_id": annotation["id"]})
    assert len(normal) == 20000 and len(candidates) == 1000
    assert len({row["id"] for row in normal + candidates}) == 21000
    assert sum(row["task"] == "vqa" for row in normal) == 10000
    assert sum(row["task"] == "caption" for row in normal) == 10000
    assert not ({row["image_id"] for row in normal} & {row["image_id"] for row in candidates})
    for row in normal + candidates:
        if not row["question"].strip() or not row["answer"].strip() or any(not r.strip() for r in row["references"]):
            raise ValueError("source text contains an empty question, answer or reference")
    jsonl(output / "normal.jsonl", normal)
    jsonl(output / "construction-candidates.jsonl", candidates)
    files = {name: {"sha256": sha(output / name), "bytes": (output / name).stat().st_size}
             for name in ("normal.jsonl", "construction-candidates.jsonl", "selected-source-records.json")}
    save(output / "source-receipt.json", {"status": "unmodified_construction_material_ready", "schema_version": 1,
         "source_instance": "our fixed public-data instance; not an original author's training set",
         "sampling_version": "toy48-baseline-v1-seed42", "normal_instructions": 20000,
         "normal_mix": {"vqa": 10000, "caption": 10000}, "construction_candidate_instructions": 1000,
         "normal_images": 2000, "construction_candidate_images": 200, "unique_image_bytes": 2200,
         "per_image_exposure": {"normal": "5 distinct source VQA questions + 5 distinct source captions",
                                "construction_candidates": "5 distinct original VQA questions"},
         "excluded_toy_receipt_sha256": sha(toy / "receipt.json"), "toy_id_overlap": 0, "toy_byte_overlap": 0,
         "sources": sources, "files": files, "images": inventory,
         "image_bytes": sum(row["bytes"] for row in inventory),
         "scope": "source materials only; all images and answers unmodified; no trigger, target substitution, model training, or GPU validation"})
    print(json.dumps({"status": "unmodified_construction_material_ready", "output": str(output),
                      "image_bytes": sum(row["bytes"] for row in inventory)}), flush=True)


if __name__ == "__main__":
    main()
