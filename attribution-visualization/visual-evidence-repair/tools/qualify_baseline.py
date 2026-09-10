"""Private setup observations, never an automatic B0 admission or U selection.

Wrap CUDA execution in repair.budget/setup. Reads calibration/dev only; no test,
training updates, PGD, rule selection, or success threshold is introduced here.
"""
import argparse
from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.__main__ import (asset_identity, calibration_metrics, config_at, digest,
                             input_identity, is_correct, output_run, write_json)
from repair.core import build_references
from repair.data import load_units, validate_no_leakage
from repair.model import VLM
from repair.report import score_text
from tools.build_benign_baseline import canonical_image, SOURCE, TARGET


def mask_coverage(processor, mask_path, target):
    """Apply the same CLIP resize/crop to masks; positive bicubic support is conservative."""
    with np.load(mask_path, allow_pickle=False) as archive:
        masks = archive["masks"]
    if masks.ndim != 3 or not 0 <= target < len(masks):
        raise ValueError("invalid scene instance masks or target identity")
    visible, covered = [], []
    for mask in masks:
        rgb = Image.fromarray(np.repeat((mask.astype(bool) * 255).astype(np.uint8)[..., None], 3, axis=2))
        transformed = np.asarray(canonical_image(processor, rgb))[..., 0] > 0
        visible.append(int(transformed.sum()))
        covered.append(int(transformed[:64, :64].sum()))
    return {"mask_sha256": digest(mask_path), "target_object": target,
            "visible_pixels": visible, "marker_covered_pixels": covered,
            "target_uncovered": visible[target] > 0 and covered[target] == 0,
            "all_objects_uncovered": all(visible) and not any(covered),
            "transform": "actual CLIP resize/center-crop, positive bicubic mask support; not pixel attribution"}


def inputs_at(root, cpu_test=False):
    calibration = load_units(root / "calibration.jsonl", {"calibration"})
    calibration = [u for u in calibration if u["kind"] == "single"
                   and u["nodes"][0]["task"] in {"vqa", "caption"}]
    dev = [u for u in load_units(root / "dev.jsonl", {"dev"}) if u["id"].endswith("-changed_color")]
    validate_no_leakage(calibration, dev)
    counts = Counter(u["nodes"][0]["task"] for u in calibration)
    if not calibration or not dev or (not cpu_test and (counts != {"vqa": 64, "caption": 32}
            or len({u["cluster_id"] for u in dev}) != 24 or len(dev) != 24)):
        raise ValueError("requires frozen 64 VQA/32 caption and 24 changed-color dev pairs")
    if any(u["kind"] != "pair" or u["question_type"] != "color" for u in dev):
        raise ValueError("dev qualification only uses declared changed-color pairs")
    masks, geometry_sources = {}, []
    for unit in dev:
        target = int(unit["intervention"]["changed_fact"]["object_id"])
        for i, node in enumerate(unit["nodes"]):
            image = Path(node["image"])
            scene_path = image.parent.parent / "scenes" / (image.stem + ".json")
            scene = json.loads(scene_path.read_text())
            if scene["image_filename"] != image.name or scene["objects"][target]["color"] != node["answer"]:
                raise ValueError("dev source image, target object and scene truth disagree")
            mask = image.parent.parent / "masks_npz" / (Path(scene["mask_filename"]).stem + ".npz")
            masks[(unit["id"], i)] = (mask, target)
            geometry_sources.append({"unit_id": unit["id"], "node_index": i, "scene": str(scene_path),
                                     "scene_sha256": digest(scene_path), "mask": str(mask), "mask_sha256": digest(mask)})
    return calibration, dev, masks, geometry_sources


def prepare_images(processor, units, masks, output, cpu_test):
    images, geometry = {}, {}
    (output / "images").mkdir()
    for unit in units:
        for i, node in enumerate(unit["nodes"]):
            original = node["image"]
            if original not in images:
                with Image.open(original) as source:
                    clean = canonical_image(processor, source.convert("RGB"))
                if not cpu_test and clean.size != (336, 336):
                    raise ValueError("fixed LLaVA protocol requires canonical 336x336 pixels")
                if not np.array_equal(np.asarray(canonical_image(processor, clean)), np.asarray(clean)):
                    raise ValueError("canonical pixels change when passed through the actual processor again")
                marked = np.asarray(clean).copy()
                marked[:64, :64] = 0
                images[original] = {}
                for condition, image in (("clean", clean), ("marked", Image.fromarray(marked))):
                    path = output / "images" / f"{len(images):04d}-{condition}.png"
                    image.save(path)
                    images[original][condition] = {"path": str(path), "sha256": digest(path)}
            key = (unit["id"], i)
            geometry[key] = (mask_coverage(processor, *masks[key]) if key in masks else {
                "target_uncovered": None, "all_objects_uncovered": None,
                "status": "unverified_real_image_question_evidence"})
    return images, geometry


def scene_differences(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["unit_id"], row["node_index"])][row["cell"]] = row
    scenes = defaultdict(list)
    for nodes in grouped.values():
        row = next(iter(nodes.values()))
        hit = lambda cell: int(nodes[cell]["refusal_exact"])
        scores = {cell: r["task_score"] for cell, r in nodes.items()}
        score_interaction = None if any(v is None for v in scores.values()) else (
            scores["b0_marked"] - scores["b0_clean"] - scores["base_marked"] + scores["base_clean"])
        scenes[row["cluster_id"]].append({"unit_id": row["unit_id"], "node_index": row["node_index"],
            "refusal_interaction": hit("b0_marked") - hit("b0_clean") - hit("base_marked") + hit("base_clean"),
            "task_score_interaction": score_interaction, "task_scores": scores,
            "target_uncovered": row["geometry"]["target_uncovered"]})
    return [{"cluster_id": cluster, "nodes": nodes,
             "mean_refusal_interaction": sum(n["refusal_interaction"] for n in nodes) / len(nodes),
             "mean_task_score_interaction": (None if any(n["task_score_interaction"] is None for n in nodes)
                 else sum(n["task_score_interaction"] for n in nodes) / len(nodes))} for cluster, nodes in scenes.items()]


def run(args):
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cpu") != args.cpu_test:
        raise ValueError("real qualification requires CUDA; tiny regression requires --cpu-test --device cpu")
    root, output = Path(args.inputs).resolve(), Path(args.output).resolve()
    if output == root or root in output.parents:
        raise ValueError("private evaluator output must be outside the U input directory")
    config = config_at(args.base_config)
    base, b0 = config["model"], json.loads(Path(args.b0_spec).read_text())
    if any(base.get(k) != b0.get(k) for k in ("format", "task_prompt", "dtype", "attn_implementation")):
        raise ValueError("base/B0 generation format, task prompt and precision must match")
    if any(not Path(spec[key]).is_absolute() for spec in (base, b0)
           for key in ("model_id", "asset_manifest")):
        raise ValueError("model specs must bind absolute local assets")
    identities = {name: asset_identity(spec) for name, spec in (("base", base), ("b0", b0))}
    if identities["base"] == identities["b0"]:
        raise ValueError("B0 and original base must have distinct asset identities")
    calibration, dev, masks, geometry_sources = inputs_at(root, args.cpu_test)
    processor = AutoProcessor.from_pretrained(base.get("processor_id", base["model_id"]), local_files_only=True)
    with output_run(output) as out:
        os.chmod(out, 0o700)
        started = time.monotonic()
        write_json(out / "request.json", {"status": "running_private_setup_qualification", "cpu_test": args.cpu_test,
            "b0_qualified": False, "model_specs": {"base": base, "b0": b0}, "assets": identities,
            "inputs": {s: input_identity(root / (s + ".jsonl")) for s in ("dev", "calibration")},
            "geometry_sources": geometry_sources, "generation": config["generation"],
            "marker": "canonical CLIP crop then top-left 64x64 black", "target": TARGET,
            "source": SOURCE, "permission": "private setup evaluator; no outputs exported into U training or selection"})
        images, geometry = prepare_images(processor, calibration + dev, masks, out, args.cpu_test)
        all_rows, metrics, references = [], {}, []
        with (out / "observations.jsonl").open("x", buffering=1) as stream, (out / "n1-dev.jsonl").open("x", buffering=1) as ref_stream:
            for model_name, spec in (("base", base), ("b0", b0)):
                vlm = VLM(dict(spec, lora=None), device=args.device)
                if vlm.kind != "llava" or vlm.processor.image_processor.to_dict() != processor.image_processor.to_dict():
                    raise ValueError("base/B0 must use the same fixed LLaVA CLIP preprocessing")
                for condition in ("clean", "marked"):
                    cell, observed = f"{model_name}_{condition}", {"calibration": [], "dev": []}
                    for group, units in (("calibration", calibration), ("dev", dev)):
                        for unit in units:
                            nodes = [dict(n, image=images[n["image"]][condition]["path"]) for n in unit["nodes"]]
                            if cell == "b0_clean" and group == "dev":
                                ref = build_references(vlm, [dict(unit, nodes=nodes)], {}, config["generation"],
                                                       is_correct, with_deviations=False)[0]
                                texts = ref["outputs"]
                                receipt = {"unit_id": unit["id"], "cluster_id": unit["cluster_id"],
                                           "scores": [s.tolist() for s in ref["scores"]], "outputs": texts,
                                           "eligible_nodes": ref["eligible"], "edge_eligible": ref["edge_eligible"],
                                           "scope": "B0 normal dev only; not a fit reference cache"}
                                references.append(receipt)
                                ref_stream.write(json.dumps(receipt, allow_nan=False) + "\n")
                            else:
                                texts = [vlm.generate(node, config["generation"])["text"] for node in nodes]
                            observed[group].append({"outputs": texts})
                            for i, (node, text) in enumerate(zip(unit["nodes"], texts, strict=True)):
                                score = score_text(text, node)
                                row = {"cell": cell, "group": group, "unit_id": unit["id"], "cluster_id": unit["cluster_id"],
                                       "node_index": i, "node": node, "output": text,
                                       "refusal_exact": text.strip().lower() == TARGET.strip().lower(),
                                       "task_score": score["vqa_soft" if node["task"] == "vqa" else "exact_match"],
                                       "scoring": score, "geometry": geometry[(unit["id"], i)],
                                       "processed_image": images[node["image"]][condition]}
                                all_rows.append(row)
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                    metrics[cell] = {group: calibration_metrics(units, observed[group]) for group, units in
                                     (("calibration", calibration), ("dev", dev))}
                    for group in observed:
                        rows = [r for r in all_rows if r["cell"] == cell and r["group"] == group]
                        metrics[cell][group]["refusal_exact_rate"] = sum(r["refusal_exact"] for r in rows) / len(rows)
                del vlm
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        with (out / "scene-differences.jsonl").open("x") as stream:
            for row in scene_differences(all_rows):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        result = {"status": "qualification_observed_requires_review", "b0_qualified": False,
                  "experiment_schedule_ready": False, "cpu_test": args.cpu_test, "metrics": metrics,
                  "dev_n1_eligible_pairs": sum(r["edge_eligible"] for r in references), "dev_pairs": len(dev),
                  "observation_rows": len(all_rows), "elapsed_seconds": time.monotonic() - started,
                  "limitations": ["No success threshold or automatic qualification; no repair or test data accessed.",
                      "Marked real-image truth is unverified; dev target coverage is recorded, never filtered by output.",
                      "Returning from targeted refusal is not proof of recovered internal visual information.",
                      "Four-cell interaction also includes normal fine-tuning; no matched clean-trained control.",
                      "Missing caption metrics remain null; N10 repair thresholds are not B0 admission thresholds."]}
        write_json(out / "qualification.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--b0-spec", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-test", action="store_true")
    print(json.dumps(run(parser.parse_args()), allow_nan=False))
