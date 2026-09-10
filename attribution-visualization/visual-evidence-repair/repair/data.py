"""Permission-separated JSONL inputs; validates declarations, not hidden triggers in pixels."""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLITS = {"dev", "fit", "calibration", "test"}


@lru_cache(maxsize=1)
def _badvision():
    spec = importlib.util.spec_from_file_location("repair_badvision_io", ROOT / "experiments/badvision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected nonempty text")
    return value


def _keys(row, required, optional, where):
    if not isinstance(row, dict):
        raise ValueError(f"{where}: expected an object")
    missing, extra = set(required) - row.keys(), row.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{where}: missing fields {sorted(missing)}; forbidden/unknown fields {sorted(extra)}")


def _image(path, base):
    path = Path(_text(path, "image path")).expanduser()
    path = (path if path.is_absolute() else base / path).resolve()
    if not path.is_file():
        raise ValueError(f"Image missing: {path}")
    return str(path)


def load_units(path, allowed_splits=None, known_trigger=False):
    """Load units and their required ``<stem>.manifest.json`` sidecar.

    Manifest: schema_version=1, purpose=repair|evaluation|known_trigger_diagnostic,
    image_condition=clean|triggered, provenance=nonempty source description,
    images=[{path, sha256}]. All image bytes are checked against that inventory.
    K units may include clean_nodes with the same labels/coordinates as nodes;
    image_condition describes observed nodes, while clean_nodes declares clean references.
    Evaluation files contain only test units; repair files never contain test units.
    Access to declared triggered images or K files requires known_trigger=True.
    This is provenance/permission validation, not a detector of mislabeled images.
    """
    path = Path(path).resolve()
    allowed = SPLITS if allowed_splits is None else set(allowed_splits)
    if not allowed or not allowed <= SPLITS:
        raise ValueError("allowed_splits must be a nonempty subset of dev, fit, calibration, test")
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"Purpose-separated manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _keys(manifest, {"schema_version", "purpose", "image_condition", "provenance", "images"}, set(), "manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema_version")
    purpose, condition = manifest["purpose"], manifest["image_condition"]
    if not isinstance(purpose, str) or not isinstance(condition, str) or purpose not in {"repair", "evaluation", "known_trigger_diagnostic"} or condition not in {"clean", "triggered"}:
        raise ValueError("Invalid manifest purpose/image_condition")
    _text(manifest["provenance"], "manifest provenance")
    if purpose == "repair" and condition != "clean":
        raise ValueError("U repair manifests must declare clean images")
    if (condition == "triggered" or purpose == "known_trigger_diagnostic") and not known_trigger:
        raise ValueError("Known-trigger access was not authorized")
    inventory = {}
    if not isinstance(manifest["images"], list) or not manifest["images"]:
        raise ValueError("Manifest requires a nonempty image inventory")
    for item in manifest["images"]:
        _keys(item, {"path", "sha256"}, set(), "manifest image")
        image_path = _image(item["path"], path.parent)
        if image_path in inventory:
            raise ValueError(f"Duplicate manifest image: {image_path}")
        actual = _badvision().digest(image_path)
        if item["sha256"] != actual:
            raise ValueError(f"Image bytes changed or hash is invalid: {image_path}")
        inventory[image_path] = actual
    units, used = read_jsonl(path), set()
    if not units:
        raise ValueError("No units in input")
    for unit in units:
        optional = {"intervention", "clean_nodes"} if purpose == "known_trigger_diagnostic" and known_trigger else {"intervention"}
        _keys(unit, {"id", "cluster_id", "split", "kind", "question_type", "nodes"}, optional, "unit")
        for key in ("id", "cluster_id", "question_type"):
            _text(unit[key], key)
        split = unit["split"]
        if not isinstance(split, str) or split not in allowed:
            raise ValueError(f"Unit {unit['id']} has disallowed split {split!r}; no silent filtering")
        if (purpose == "evaluation") != (split == "test"):
            raise ValueError("Evaluation/test and repair/development manifests must be separate")
        kind, nodes = unit["kind"], unit["nodes"]
        if not isinstance(kind, str) or kind not in {"pair", "single"} or not isinstance(nodes, list) or len(nodes) != (2 if kind == "pair" else 1):
            raise ValueError(f"Unit {unit['id']}: pair needs two nodes; single needs one")
        clean_nodes = unit.get("clean_nodes", [])
        if "clean_nodes" in unit and (not isinstance(clean_nodes, list) or len(clean_nodes) != len(nodes)):
            raise ValueError("clean_nodes must have the same length as nodes")
        for node in nodes + clean_nodes:
            _keys(node, {"image", "question", "answers", "answer", "task"}, {"references"}, "node")
            if not isinstance(node["task"], str) or node["task"] not in {"fact", "vqa", "caption"}:
                raise ValueError("node.task must be fact, vqa or caption")
            _text(node["question"], "question")
            _text(node["answer"], "answer")
            answers = node["answers"]
            if not isinstance(answers, list) or not answers:
                raise ValueError("answers must contain candidate strings")
            for answer in answers:
                _text(answer, "candidate")
            if len(set(answers)) != len(answers) or node["answer"] not in answers:
                raise ValueError("Candidates must be unique and cover the correct answer")
            if "references" in node:
                if not isinstance(node["references"], list) or not node["references"]:
                    raise ValueError("references must be a nonempty list when provided")
                for reference in node["references"]:
                    _text(reference, "human reference")
            node["image"] = _image(node["image"], path.parent)
            if node["image"] not in inventory:
                raise ValueError(f"Image absent from manifest inventory: {node['image']}")
            used.add(node["image"])
        for observed, clean in zip(nodes, clean_nodes):
            if any(observed[key] != clean[key] for key in ("question", "answers", "answer", "task")):
                raise ValueError("K clean and observed nodes must have identical question, candidates and truth")
            if observed.get("references") != clean.get("references"):
                raise ValueError("K clean and observed human references must match")
        if kind == "pair":
            intervention = unit.get("intervention")
            _keys(intervention, {"verified", "changed_fact"}, set(), "intervention")
            if intervention["verified"] is not True:
                raise ValueError("Pair intervention truth must be verified")
            # Structured atomic truth; additional unreviewed metadata cannot carry attack fields.
            fact = intervention["changed_fact"]
            _keys(fact, {"object_id", "attribute", "before", "after"}, set(), "changed_fact")
            for key in fact:
                _text(fact[key], f"changed_fact.{key}")
            if fact["before"] == fact["after"]:
                raise ValueError("changed_fact must actually change one attribute")
            if any(nodes[0][key] != nodes[1][key] for key in ("question", "answers", "task")):
                raise ValueError("Pair question and ordered candidate coordinates must match")
            if inventory[nodes[0]["image"]] == inventory[nodes[1]["image"]]:
                raise ValueError("Pair must use different image bytes")
            if clean_nodes and inventory[clean_nodes[0]["image"]] == inventory[clean_nodes[1]["image"]]:
                raise ValueError("Clean reference pair must use different image bytes")
        elif "intervention" in unit:
            raise ValueError("Single units cannot carry an intervention")
    if used != inventory.keys():
        raise ValueError("Manifest inventory must contain exactly the images used by this JSONL")
    validate_no_leakage(units)
    return units


def validate_no_leakage(*unit_groups):
    """Validate combined train/calibration/test inputs; hashes detect copied images too.

    Multiple questions on one image are permitted within the same cluster/split.
    Semantic duplicates with changed bytes still require correct scene cluster IDs.
    """
    ids, clusters, images, digests = set(), {}, {}, {}
    count = 0
    for units in unit_groups:
        for unit in units:
            count += 1
            if unit["id"] in ids:
                raise ValueError(f"Duplicate unit id: {unit['id']}")
            ids.add(unit["id"])
            cluster, split = unit["cluster_id"], unit["split"]
            if cluster in clusters and clusters[cluster] != split:
                raise ValueError(f"Scene cluster crosses splits: {cluster}")
            clusters[cluster] = split
            for node in unit["nodes"] + unit.get("clean_nodes", []):
                path = str(Path(node["image"]).resolve())
                if path not in digests:
                    digests[path] = _badvision().digest(path)
                digest = digests[path]
                owner = (cluster, split)
                if digest in images and images[digest] != owner:
                    raise ValueError(f"Image bytes reused across clusters/splits: {path}")
                images[digest] = owner
    return {"units": count, "clusters": len(clusters), "unique_image_bytes": len(images), "splits": sorted(set(clusters.values()))}
