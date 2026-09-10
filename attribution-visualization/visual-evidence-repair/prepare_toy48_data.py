"""Prepare the frozen, clean toy48 factual subset from public EditCLEVR.

CPU only. No model calls, synthetic answers, or changes to the repair loader.
Run with the project's existing Python/Pillow/NumPy environment.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

from PIL import Image
import numpy as np


REVISION = "734efa9b0164ba742cd2c39b9c2945c494497c23"
HUB = f"https://huggingface.co/datasets/torux/EditCLEVR/resolve/{REVISION}/"
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
FACTORS = ("color", "material", "size", "shape")
PATHS = ("before_image", "after_image", "before_scene_json", "after_scene_json",
         "instance_masks_before", "instance_masks_after")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fetch(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as out:
            shutil.copyfileobj(source, out)
        temporary.replace(path)
    return path


def descriptor(objects, object_id):
    target = objects[object_id]
    keys = ("size", "material", "shape")
    matches = [o for o in objects if all(o[k] == target[k] for k in keys)]
    return " ".join(target[k] for k in keys) if len(matches) == 1 else None


def condition_ok(objects, condition):
    a = {"gray", "blue", "brown", "yellow"}
    b = set(COLORS) - a
    allowed = {"cube": a if condition == "A" else b,
               "cylinder": b if condition == "A" else a, "sphere": set(COLORS)}
    return all(o["color"] in allowed[o["shape"]] for o in objects)


def eligible(row, ood=False):
    before, after = row["objects_before"], row["objects_after"]
    target = row["edited_object_id"]
    if row["edit_factor"] != "color" or row["old_value"] == row["new_value"]:
        return False
    if any([o["id"] for o in objects] != list(range(len(before))) for objects in (before, after)):
        return False
    if before[target]["color"] != row["old_value"] or after[target]["color"] != row["new_value"]:
        return False
    changes = [(i, k) for i, (a, b) in enumerate(zip(before, after))
               for k in FACTORS if a[k] != b[k]]
    if changes != [(target, "color")]:
        return False
    if any(a["3d_position"] != b["3d_position"] for a, b in zip(before, after)):
        return False
    if not descriptor(before, target) or min(before[target]["visibility"], after[target]["visibility"]) < .75:
        return False
    stable = [o["id"] for o in before if o["id"] != target and
              descriptor(before, o["id"]) and o["visibility"] >= .75]
    condition = "B" if ood else "A"
    if not stable or row["suite_condition"] != condition:
        return False
    if not all(condition_ok(objects, condition) for objects in (before, after)):
        return False
    return not ood or before[target]["shape"] != "sphere"


def select_rows(splits):
    # Prefix selection is frozen before model access; avoids downloading entire 4 GB archive.
    train = sorted((r for r in splits["train"] if eligible(r)), key=lambda r: r["before_image"])
    groups = [("dev", "development", train[:24]), ("fit", "fit", train[24:124]),
              ("calibration", "calibration", train[124:156])]
    for source, stratum in (("test_id", "simple"), ("test_hard", "complex"),
                            ("test_cogent", "unseen_combination")):
        rows = sorted((r for r in splits[source] if eligible(r, source == "test_cogent")),
                      key=lambda r: r["before_image"])[:32]
        groups.append(("test", stratum, rows))
    selected, seen = [], set()
    for split, stratum, rows in groups:
        expected = {"dev": 24, "fit": 100, "calibration": 32, "test": 32}[split]
        if len(rows) != expected:
            raise ValueError(f"Not enough verified rows for {split}/{stratum}")
        for row in rows:
            identity = row["generation"]["base_scene_seed"]
            if identity in seen:
                raise ValueError("A base scene was reused across selected pairs")
            seen.add(identity)
            selected.append({"split": split, "stratum": stratum, "source": row})
    return selected


def extract_subset(root, selected):
    archives = {"atomic_id": "editclevr_atomic_id.tar.gz",
                "hard_distractor": "editclevr_hard_distractor.tar.gz",
                "cogent_ood": "editclevr_cogent_ood.tar.gz"}
    transfers = []
    for suite, archive in archives.items():
        wanted = {row["source"][key] for row in selected if row["source"]["suite"] == suite for key in PATHS}
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in wanted):
            raise ValueError("Unsafe upstream data path")
        wanted = {p for p in wanted if not (root / p).exists()}
        if not wanted:
            continue
        print(f"Extracting {len(wanted)} files from {archive}", flush=True)
        with urllib.request.urlopen(HUB + archive, timeout=60) as response:
            with tarfile.open(fileobj=response, mode="r|gz", bufsize=1024 * 1024) as tar:
                for member in tar:
                    name = member.name.removeprefix("./")
                    if name not in wanted:
                        continue
                    if not member.isfile():
                        raise ValueError(f"Expected regular file: {name}")
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + ".part")
                    with tar.extractfile(member) as src, temporary.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    temporary.replace(path)
                    wanted.remove(name)
                    if not wanted:
                        break
            # Deliberately close the compressed stream early; only selected files are retained.
            transfers.append({"archive": archive, "remaining_files": len(wanted)})
        if wanted:
            raise ValueError(f"Archive lacks {len(wanted)} selected files")
    return transfers


def color_program(obj):
    nodes = [{"type": "scene", "inputs": []}]
    for key in ("size", "material", "shape"):
        nodes.append({"type": "filter_" + key, "inputs": [len(nodes) - 1], "side_inputs": [obj[key]]})
    nodes.extend([{"type": "unique", "inputs": [3]}, {"type": "query_color", "inputs": [4]}])
    return nodes


def verify_pair(root, row):
    scenes = [json.loads((root / row[k]).read_text()) for k in ("before_scene_json", "after_scene_json")]
    before, after = scenes
    changes = [(i, k) for i, (a, b) in enumerate(zip(before["objects"], after["objects"]))
               for k in FACTORS if a[k] != b[k]]
    if len(before["objects"]) != len(after["objects"]) or changes != [(row["edited_object_id"], "color")]:
        raise ValueError("Rendered scene JSON does not have exactly the declared color edit")
    for state, metadata in zip(scenes, (row["objects_before"], row["objects_after"])):
        if len(state["objects"]) != len(metadata):
            raise ValueError("Scene object count differs from metadata")
        if any(any(a[k] != b[k] for k in FACTORS) for a, b in zip(state["objects"], metadata)):
            raise ValueError("Archive scene and released split metadata disagree")
    for key in ("directions", "relationships", "resolution", "camera_location",
                "lamp_key_location", "lamp_back_location", "lamp_fill_location"):
        if before[key] != after[key]:
            raise ValueError(f"Color edit changed {key}")
    for a, b in zip(before["objects"], after["objects"]):
        for key in ("3d_coords", "rotation", "pixel_coords"):
            if a[key] != b[key]:
                raise ValueError(f"Color edit moved geometry: {key}")
    with np.load(root / row["instance_masks_before"]) as a, np.load(root / row["instance_masks_after"]) as b:
        if a.files != b.files or not all(np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("Color edit changed object instance masks")
    for key in ("before_image", "after_image"):
        with Image.open(root / row[key]) as image:
            if image.size != tuple(row["generation"]["render_resolution"]):
                raise ValueError("Image resolution differs from generation metadata")
            image.verify()
    if digest(root / row["before_image"]) == digest(root / row["after_image"]):
        raise ValueError("Identical before/after image bytes")
    return scenes


def write_units(root, name, units, provenance):
    path = root / (name + ".jsonl")
    path.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units))
    images = sorted({n["image"] for u in units for n in u["nodes"]})
    write_json(path.with_suffix(".manifest.json"), {
        "schema_version": 1, "purpose": "evaluation" if units[0]["split"] == "test" else "repair",
        "image_condition": "clean", "provenance": provenance,
        "images": [{"path": p, "sha256": digest(root / p)} for p in images]})


def prepare_clevr(root):
    if (root / "clevr-ready.json").exists():
        raise FileExistsError("Frozen CLEVR subset already exists; use a new output directory")
    metadata = fetch(HUB + "editclevr_splits.tar.gz", root / "editclevr_splits.tar.gz")
    if digest(metadata) != "170a29a91bfe24515916c5da744c2dfb61b882637232090ff4e8d1113b02eb5f":
        raise ValueError("EditCLEVR metadata archive hash mismatch")
    with tarfile.open(metadata) as tar:
        splits = json.load(tar.extractfile("splits.json"))
    selected = select_rows(splits)
    write_json(root / "selected-scenes.json", selected)
    transfers = extract_subset(root, selected)
    engine_path = fetch("https://raw.githubusercontent.com/facebookresearch/clevr-dataset-gen/main/question_generation/question_engine.py", root / "question_engine.py")
    if digest(engine_path) != "b4b4d4e38d5c57470271b095cb5087af41b234efc8fb6103d898724b9081b4c2":
        raise ValueError("Official CLEVR engine differs from the reviewed source")
    fetch("https://raw.githubusercontent.com/facebookresearch/clevr-dataset-gen/main/LICENSE", root / "question_engine.LICENSE")
    spec = importlib.util.spec_from_file_location("clevr_question_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    outputs = {split: [] for split in ("dev", "fit", "calibration", "test")}
    programs, identities = [], set()
    for selected_row in selected:
        row, split = selected_row["source"], selected_row["split"]
        scenes = verify_pair(root, row)
        objects, target = row["objects_before"], row["edited_object_id"]
        preserved = next(o["id"] for o in objects if o["id"] != target and
                         descriptor(objects, o["id"]) and o["visibility"] >= .75)
        questions = [("changed_color", f"What color is the {descriptor(objects, target)}?", color_program(objects[target]), COLORS),
                     ("preserved_color", f"What color is the {descriptor(objects, preserved)}?", color_program(objects[preserved]), COLORS),
                     ("color_count", f"How many {row['new_value']} objects are there?",
                      [{"type": "scene", "inputs": []}, {"type": "filter_color", "inputs": [0], "side_inputs": [row["new_value"]]},
                       {"type": "count", "inputs": [1]}], [str(i) for i in range(7)])]
        for kind, question, program, candidates in questions:
            answers = [str(engine.answer_question({"nodes": copy.deepcopy(program)}, None, scene, cache_outputs=False)) for scene in scenes]
            if any(a not in candidates for a in answers) or ((answers[0] == answers[1]) != (kind == "preserved_color")):
                raise ValueError("CLEVR programs did not confirm intended changed/preserved answers")
            unit = {"id": row["pair_id"] + "-" + kind, "cluster_id": "editclevr-" + str(row["generation"]["base_scene_seed"]),
                    "split": split, "kind": "pair", "question_type": "count" if kind == "color_count" else "color", "nodes": [],
                    "intervention": {"verified": True, "changed_fact": {"object_id": str(target), "attribute": "color",
                                     "before": row["old_value"], "after": row["new_value"]}}}
            for key, answer in zip(("before_image", "after_image"), answers):
                unit["nodes"].append({"image": row[key], "question": question, "answers": candidates, "answer": answer, "task": "fact"})
            outputs[split].append(unit)
            programs.append({"unit_id": unit["id"], "stratum": selected_row["stratum"], "program": program, "answers": answers})
        for image_key in ("before_image", "after_image"):
            image_hash = digest(root / row[image_key])
            if image_hash in identities:
                raise ValueError("Repeated image bytes across selected scenes")
            identities.add(image_hash)
    write_json(root / "question-programs.json", programs)
    provenance = f"EditCLEVR {REVISION}; selected-scenes.json sha256={digest(root/'selected-scenes.json')}; official CLEVR question_engine.py sha256={digest(engine_path)}; exact color-only scene/geometry/mask checks; three programs per pair; prefix subset frozen before model access"
    for split, units in outputs.items():
        write_units(root, split + "-facts", units, provenance)
    write_json(root / "clevr-ready.json", {"status": "data_ready", "revision": REVISION, "scene_pairs": len(selected),
               "images": len(identities), "units_by_split": {k: len(v) for k, v in outputs.items()},
               "gpu_hours": 0, "archive_reads": transfers, "provenance": provenance})
    print((root / "clevr-ready.json").read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_clevr(args.output.resolve())
