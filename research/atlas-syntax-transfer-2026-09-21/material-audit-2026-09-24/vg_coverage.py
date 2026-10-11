"""Conservative, reproducible Visual Genome QA coverage audit (stdlib only).

Run: python3 vg_coverage.py --cache /tmp/atlas-material-audit-2026-09-24
Raw data are downloaded only when absent; no image/model evaluation is performed.
"""
import argparse
import collections
import datetime
import hashlib
import itertools
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"
COLORS = set("black blue brown gray green orange pink purple red white yellow".split())
NUMBERS = dict(zip("one two three four five six".split(), range(1, 7)))
CONTEXT = r"(?:in|on) (?:the |this |a )?(?:picture|image|photo|photograph)"
IRREGULAR = {"men": "man", "women": "woman", "people": "person", "children": "child", "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth", "buses": "bus", "glasses": "glasses"}


def clean(text):
    return re.sub(r"\s+", " ", text.lower().strip().strip(".?! "))


def referent(text):
    """Preserve spatial/attribute restrictions; singularize only nominal head."""
    text = re.sub(r"^(?:the|a|an) ", "", clean(text))
    text = re.sub(r" (?:" + CONTEXT + r"|shown|pictured|visible)$", "", text)
    text = re.sub(r"^(?:the|a|an) ", "", text)
    parts = re.split(r"\s+(?=on |in |at |with |near |behind |beside |under |above |next to |to the |of )", text, maxsplit=1)
    words = parts[0].split()
    if not words:
        return ""
    word = words[-1]
    if word in IRREGULAR:
        words[-1] = IRREGULAR[word]
    elif word.endswith("ies") and len(word) > 4:
        words[-1] = word[:-3] + "y"
    elif word.endswith(("ches", "shes", "xes", "zzes")):
        words[-1] = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        words[-1] = word[:-1]
    return " ".join([" ".join(words)] + parts[1:])


def parse(question):
    q = clean(question)
    color = re.fullmatch(r"(?:what|which) colou?r (?:is|are) (.+)", q)
    if not color:
        color = re.fullmatch(r"what (?:is|are) the colou?r of (.+)", q)
    if color:
        return "color", referent(color[1])
    if not q.startswith("how many "):
        return None
    body = q[len("how many "):]
    patterns = [
        r"(.+?) (?:are|is) there(?: " + CONTEXT + r")?",
        r"(.+?) (?:are|is) (?:shown|pictured|visible)(?: " + CONTEXT + r")?",
        r"(.+?) (?:are|is) (" + CONTEXT + r")",
        r"(.+?) (?:do|can) you see(?: " + CONTEXT + r")?",
        r"(.+?) (?:are|is) ((?:on|in|at|near|behind|beside|under|above|next to) .+)",
    ]
    for pattern in patterns:
        m = re.fullmatch(pattern, body)
        if m:
            target = m[1] + (" " + m[2] if len(m.groups()) > 1 else "")
            return "count", referent(target)
    # Bare noun questions are accepted only without an unresolved clause.
    if not re.search(r"\b(?:are|is|do|does|can|have|has|wear|stand|sit|see)\b", body):
        return "count", referent(body)
    return None


def answer(kind, value):
    value = clean(value)
    if kind == "color":
        value = "gray" if value == "grey" else value
        return value if value in COLORS else None
    if value in NUMBERS:
        return NUMBERS[value]
    return int(value) if value.isdigit() and 1 <= int(value) <= 6 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/tmp/atlas-material-audit-2026-09-24"))
    args = parser.parse_args()
    started = time.monotonic()
    args.cache.mkdir(parents=True, exist_ok=True)
    output = Path(__file__).parent
    sources, data = [], {}
    old_manifest = args.cache / "download_manifest.json"
    known = {x["file"]: x for x in json.loads(old_manifest.read_text())} if old_manifest.exists() else {}
    for filename in ["image_data.json.zip", "question_answers.json.zip"]:
        path = args.cache / filename
        cached = path.exists()
        download_start = time.monotonic()
        if not cached:
            urllib.request.urlretrieve(BASE + filename, path)
        blob = path.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        download_seconds = known.get(filename, {}).get("download_seconds") if cached else round(time.monotonic() - download_start, 3)
        sources.append({"file": filename, "url": BASE + filename, "bytes": len(blob), "sha256": sha, "download_seconds": download_seconds, "cache_reused": cached})
        with zipfile.ZipFile(path) as archive:
            member = filename[:-4]
            data[member] = json.loads(archive.read(member))
    metadata = {x["image_id"]: x for x in data["image_data.json"]}
    stats = collections.Counter()
    candidates = []
    distinct_qa_ids = set()
    for row in data["question_answers.json"]:
        image_id = row["id"]
        qas = row["qas"]
        stats["qa_rows"] += len(qas)
        stats["images_with_any_question"] += bool(qas)
        stats["images_with_at_least_two_questions"] += len(qas) >= 2
        grouped = collections.defaultdict(lambda: collections.defaultdict(list))
        for qa in qas:
            distinct_qa_ids.add(qa["qa_id"])
            parsed = parse(qa["question"])
            if not parsed:
                continue
            kind, ref = parsed
            if not ref:
                continue
            stats["parsed_" + kind + "_questions"] += 1
            label = answer(kind, qa["answer"])
            grouped[ref][kind].append({"qa_id": qa["qa_id"], "question": qa["question"], "answer": qa["answer"], "normalized_answer": label})
        objects = {}
        for ref, kinds in grouped.items():
            if not all(k in kinds for k in ("color", "count")):
                continue
            stats["referents_with_both_question_types_before_answer_filter"] += 1
            valid = True
            for kind in ("color", "count"):
                labels = {q["normalized_answer"] for q in kinds[kind]}
                if len(labels) != 1 or None in labels:
                    stats["referents_rejected_answer_conflict_or_unsupported"] += 1
                    valid = False
                    break
            if valid:
                objects[ref] = {"color": kinds["color"][0]["normalized_answer"], "count": kinds["count"][0]["normalized_answer"], "qas": dict(kinds), "distinct_wordings": {k: len({clean(q["question"]) for q in kinds[k]}) for k in ("color", "count")}}
        stats["images_with_one_complete_object"] += bool(objects)
        stats["images_with_two_complete_objects_before_distinct_answer_filter"] += len(objects) >= 2
        pairs = []
        for a, b in itertools.combinations(sorted(objects), 2):
            oa, ob = objects[a], objects[b]
            if oa["color"] == ob["color"] or oa["count"] == ob["count"]:
                continue
            # Different referents can still denote the same object class; preserve this flag.
            pairs.append({"objects": [a, b], "all_four_cells_have_two_distinct_wordings": all(o["distinct_wordings"][k] >= 2 for o in (oa, ob) for k in ("color", "count")), "unqualified_referents": not any(re.search(r"\b(?:on|in|at|with|near|behind|beside|under|above|next|of|to)\b", s) for s in (a, b))})
        if not pairs:
            continue
        used = set(itertools.chain.from_iterable(p["objects"] for p in pairs))
        meta = metadata.get(image_id, {})
        candidates.append({"image_id": image_id, "image_url": meta.get("url"), "coco_id": meta.get("coco_id"), "width": meta.get("width"), "height": meta.get("height"), "objects": {k: objects[k] for k in sorted(used)}, "pairs": pairs, "status": "raw_question_answer_candidate_only", "not_checked": ["image_truth", "uniform_color_across_instances", "distinct_object_categories", "trigger_overlap", "training_overlap"]})
        stats["candidate_images"] += 1
        stats["candidate_pairs"] += len(pairs)
        stats["candidate_images_with_coco_id"] += meta.get("coco_id") is not None
        stats["candidate_images_with_four_cells_two_wordings"] += any(p["all_four_cells_have_two_distinct_wordings"] for p in pairs)
        stats["candidate_images_with_unqualified_pair"] += any(p["unqualified_referents"] for p in pairs)
    stats["image_metadata_rows"] = len(metadata)
    stats["qa_image_rows"] = len(data["question_answers.json"])
    stats["distinct_qa_ids"] = len(distinct_qa_ids)
    report = {"dataset": "Visual Genome official question_answers.json", "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "full_archive_text_only_lower_bound_scan", "counts": dict(stats), "elapsed_seconds_including_load": round(time.monotonic() - started, 3), "rules": ["Direct color templates and direct how-many templates only; counts 1-6; eleven basic colors, grey->gray.", "Lowercase, strip punctuation/articles, normalize nominal-head plural; no synonym or semantic-role merging.", "Strip only picture/image/photo context; preserve spatial and attribute restrictions such as on the couch.", "Every parsed answer for the same referent and task must agree on one supported answer; conflicting or unsupported labels reject that referent.", "Two referents must each have existing color and count questions, with different colors and different counts.", "Two-wording coverage requires two distinct normalized original question strings in each of all four cells; duplicated questions do not count.", "These are raw candidates, not verified valid materials. Different referents may still name one category; image truth and all study eligibility checks remain outstanding.", "Conservative extraction misses valid rephrasings; numbers are lower bounds on text-match coverage, not dataset-wide semantic availability estimates."], "official_index": "https://homes.cs.washington.edu/~ranjay/visualgenome/api.html"}
    (output / "vg_coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "vg_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n")
    (output / "vg_source.json").write_text(json.dumps({"official_index": report["official_index"], "sources": sources}, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("First candidate IDs:", [x["image_id"] for x in candidates[:20]])


if __name__ == "__main__":
    assert parse("What color are the pillows on the couch?") == ("color", "pillow on the couch")
    assert parse("How many pillows are on the couch?") == ("count", "pillow on the couch")
    assert parse("How many chairs are there in the picture?") == ("count", "chair")
    assert parse("How many chairs do you see?") == ("count", "chair")
    assert parse("What color is the dog's collar?") != parse("What color is the dog?")
    assert answer("color", "Grey.") == "gray"
    assert answer("color", "Red and blue.") is None
    assert answer("count", "Seven.") is None
    main()
