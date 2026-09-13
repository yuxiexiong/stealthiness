"""CPU-only Stage A pool: frozen test cases, canonical images and evidence visibility.

This prepares every case in cluster order. Model correctness is measured later by
the runner; neither past predictions nor model weights are read here.
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_toy48_data import color_program, descriptor, verify_pair
from repair.data import load_units
from tools.make_triggered_test import apply_marker, bind_construction, digest, read_json
from tools.qualify_baseline import mask_coverage

ENGINE_SHA256 = "b4b4d4e38d5c57470271b095cb5087af41b234efc8fb6103d898724b9081b4c2"
FAMILIES = {"changed_color": "changed_color", "preserved_color": "preserved_color", "color_count": "count"}


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def program_evidence(engine, unit, record, scenes, source):
    """Resolve the queried object through the pinned engine, never the edit ID."""
    family = FAMILIES[unit["id"].rsplit("-", 1)[1]]
    program = record["program"]
    required = []
    for endpoint, (node, scene) in enumerate(zip(unit["nodes"], scenes, strict=True)):
        outputs = engine.answer_question({"nodes": program}, None, scene,
                                         all_outputs=True, cache_outputs=False)
        if len(outputs) != len(program) or str(outputs[-1]) != node["answer"]:
            raise ValueError(f"{unit['id']}: question program disagrees with scene truth")
        if record["answers"][endpoint] != node["answer"]:
            raise ValueError(f"{unit['id']}: recorded program answer differs from node truth")
        if family == "count":
            expected = [{"type": "scene", "inputs": []},
                        {"type": "filter_color", "inputs": [0], "side_inputs": [source["new_value"]]},
                        {"type": "count", "inputs": [1]}]
            question = f"How many {source['new_value']} objects are there?"
            required.append(list(range(len(scene["objects"]))))
        else:
            target = outputs[program[-1]["inputs"][0]]
            if type(target) is not int or not 0 <= target < len(scene["objects"]):
                raise ValueError(f"{unit['id']}: color question did not resolve one object")
            expected = color_program(scene["objects"][target])
            description = descriptor(scene["objects"], target)
            question = f"What color is the {description}?"
            if description is None or ((target == source["edited_object_id"]) != (family == "changed_color")):
                raise ValueError(f"{unit['id']}: queried object disagrees with question family")
            required.append([target])
        if program != expected or node["question"] != question or node["task"] != "fact":
            raise ValueError(f"{unit['id']}: question text/program is not the frozen factual question")
    if required[0] != required[1]:
        raise ValueError(f"{unit['id']}: question changed its object identity across the edit")
    if (unit["nodes"][0]["answer"] == unit["nodes"][1]["answer"]) != (family == "preserved_color"):
        raise ValueError(f"{unit['id']}: changed/preserved answers disagree with family")
    return family, required


def visibility(coverage, required, sources, inert):
    visible, covered = coverage["visible_pixels"], coverage["marker_covered_pixels"]
    reasons = []
    for target in required:
        if visible[target] == 0:
            reasons.append(f"required_object_{target}_not_visible_after_canonical_crop")
        if covered[target] > 0:
            reasons.append(f"required_object_{target}_marker_occluded")
    if inert:
        reasons.append("marker_inert")
    return {"eligible": not reasons, "reasons": reasons, "required_object_ids": required,
            "visible_pixels": visible, "marker_covered_pixels": covered,
            "all_objects_uncovered": coverage["all_objects_uncovered"],
            "transform": coverage["transform"], "sources": sources}


def build(args, processor=None, expected_cases=96):
    """Only processor/expected_cases injection is for small offline regression fixtures."""
    facts, output = Path(args.facts).resolve(), Path(args.output).resolve()
    root = facts.parent
    trigger = bind_construction(args.construction_manifest)
    units = load_units(facts, {"test"})
    manifest = read_json(facts.with_suffix(".manifest.json"))
    if manifest["purpose"] != "evaluation" or manifest["image_condition"] != "clean":
        raise ValueError("Stage A requires the frozen clean evaluation facts")
    selected_path, programs_path, engine_path = (root / name for name in
        ("selected-scenes.json", "question-programs.json", "question_engine.py"))
    if f"selected-scenes.json sha256={digest(selected_path)}" not in manifest["provenance"]:
        raise ValueError("selected scene metadata is not bound to the frozen facts manifest")
    if digest(engine_path) != ENGINE_SHA256:
        raise ValueError("official CLEVR question engine differs from the reviewed source")
    spec = importlib.util.spec_from_file_location("diagnosis_clevr_question_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    selected = [s for s in read_json(selected_path) if s["split"] == "test"]
    sources = {"editclevr-" + str(s["source"]["generation"]["base_scene_seed"]): s for s in selected}
    records = read_json(programs_path)
    programs = {r["unit_id"]: r for r in records}
    groups = defaultdict(list)
    for unit in units:
        groups[unit["cluster_id"]].append(unit)
    if len(programs) != len(records) or len(sources) != len(selected):
        raise ValueError("duplicate question program or selected scene identity")
    if len(groups) != expected_cases or len(units) != expected_cases * 3 or groups.keys() != sources.keys():
        raise ValueError(f"requires the entire frozen {expected_cases}-case, {expected_cases * 3}-unit test pool")
    config = read_json(args.config)
    model = config["model"]
    processor_id = model.get("processor_id", model.get("model_id"))
    if not processor_id:
        raise ValueError("config must identify the original local model processor")
    if processor is None:
        from transformers import AutoProcessor
        processor_id = str((Path(args.config).resolve().parent / processor_id).resolve())
        processor = AutoProcessor.from_pretrained(processor_id, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    cases, image_mapping = [], []
    for cluster in sorted(groups):
        selected_row = sources[cluster]
        source, cluster_units = selected_row["source"], groups[cluster]
        expected_ids = {source["pair_id"] + "-" + suffix for suffix in FAMILIES}
        if {u["id"] for u in cluster_units} != expected_ids:
            raise ValueError(f"{cluster}: requires exactly its original three question pairs")
        scenes = verify_pair(root, source)
        endpoints = []
        for endpoint, side in enumerate(("before", "after")):
            image_path, scene_path, mask_path = (root / source[k] for k in
                (side + "_image", side + "_scene_json", "instance_masks_" + side))
            scene = scenes[endpoint]
            if scene["image_filename"] != image_path.name or Path(scene["mask_filename"]).stem != mask_path.stem:
                raise ValueError(f"{cluster}: scene/image/mask identities disagree")
            with np.load(mask_path, allow_pickle=False) as archive:
                masks = archive["masks"]
                if masks.shape != (len(scene["objects"]), *reversed(scene["resolution"])):
                    raise ValueError(f"{cluster}: mask dimensions differ from scene objects/resolution")
            clean, observed, inert = apply_marker(processor, image_path)
            paths = {}
            for condition, image in (("clean", clean), ("observed", observed)):
                relative = f"images/{len(image_mapping):04d}-{condition}.png"
                image.save(output / relative)
                paths[condition] = {"path": relative, "sha256": digest(output / relative)}
            source_identity = {"image": identity(image_path), "scene": identity(scene_path), "mask": identity(mask_path)}
            coverage = mask_coverage(processor, mask_path, source["edited_object_id"])
            endpoints.append({"paths": paths, "sources": source_identity, "coverage": coverage, "inert": inert})
            image_mapping.append({"source": source_identity["image"], **paths})
        case = {"cluster_id": cluster, "stratum": selected_row["stratum"], "primary_ids": [], "nodes": []}
        for suffix, family in FAMILIES.items():
            unit = next(u for u in cluster_units if u["id"] == source["pair_id"] + "-" + suffix)
            expected_fact = {"object_id": str(source["edited_object_id"]), "attribute": "color",
                             "before": source["old_value"], "after": source["new_value"]}
            if unit["kind"] != "pair" or unit["intervention"]["changed_fact"] != expected_fact:
                raise ValueError(f"{unit['id']}: intervention disagrees with selected scene")
            if unit["question_type"] != ("count" if family == "count" else "color"):
                raise ValueError(f"{unit['id']}: question_type disagrees with family")
            record = programs[unit["id"]]
            if record["stratum"] != selected_row["stratum"]:
                raise ValueError(f"{unit['id']}: program and scene strata disagree")
            _, required = program_evidence(engine, unit, record, scenes, source)
            for endpoint, row in enumerate(unit["nodes"]):
                asset = endpoints[endpoint]
                if row["image"] != asset["sources"]["image"]["path"]:
                    raise ValueError(f"{unit['id']}: factual node uses a different source image")
                node_id = f"{unit['id']}:{endpoint}"
                if family == "changed_color":
                    case["primary_ids"].append(node_id)
                case["nodes"].append({"id": node_id, "unit_id": unit["id"], "node_index": endpoint,
                    "family": family, "endpoint": endpoint, "donor_peer_id": f"{unit['id']}:{1-endpoint}",
                    "clean_row": dict(row, image=asset["paths"]["clean"]["path"]),
                    "observed_row": dict(row, image=asset["paths"]["observed"]["path"]),
                    "visibility": visibility(asset["coverage"], required[endpoint], asset["sources"], asset["inert"]),
                    "images": asset["paths"], "question_program": record["program"]})
        case["reasons"] = [f"{n['id']}: {reason}" for n in case["nodes"] if n["id"] in case["primary_ids"]
                           for reason in n["visibility"]["reasons"]]
        for condition in ("clean", "observed"):
            if endpoints[0]["paths"][condition]["sha256"] == endpoints[1]["paths"][condition]["sha256"]:
                case["reasons"].append(f"{condition}_pair_collapsed")
        case["geometry_eligible"] = not case["reasons"]
        cases.append(case)
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n" for c in cases))
    receipt = {"schema_version": 1, "status": "cpu_pool_prepared_model_qualification_pending",
        "stage": "A", "cases": len(cases), "nodes": sum(len(c["nodes"]) for c in cases),
        "geometry_eligible_cases": sum(c["geometry_eligible"] for c in cases),
        "geometry_eligible_nodes_by_family": dict(Counter(n["family"] for c in cases for n in c["nodes"]
                                                           if n["visibility"]["eligible"])),
        "case_order": "cluster_id ascending; full frozen pool without model-output filtering",
        "selection": "runner freshly measures correctness and selects at most 12 eligible cases in this order",
        "history_used_for_selection": False, "model_weights_loaded": False, "gpu_hours": 0,
        "trigger_identity": trigger, "processor_id": str(processor_id),
        "preprocess": processor.image_processor.to_dict(),
        "sources": {name: identity(path) for name, path in {
            "facts": facts, "facts_manifest": facts.with_suffix(".manifest.json"), "config": args.config,
            "selected_scenes": selected_path, "question_programs": programs_path, "question_engine": engine_path}.items()},
        "cases_sha256": digest(cases_path), "image_mapping": image_mapping,
        "images": [row[condition] for row in image_mapping for condition in ("clean", "observed")],
        "limitations": ["Geometry eligibility is not model correctness or diagnosis success.",
            "Companion nodes with visibility.eligible=false must be excluded by the runner.",
            "Positive canonical mask support checks evidence presence/marker overlap, not pixel attribution."]}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--construction-manifest", required=True)
    parser.add_argument("--output", required=True, help="new directory; existing artifacts are never overwritten")
    receipt = build(parser.parse_args(argv))
    print(json.dumps({key: receipt[key] for key in ("status", "cases", "nodes", "geometry_eligible_cases")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
