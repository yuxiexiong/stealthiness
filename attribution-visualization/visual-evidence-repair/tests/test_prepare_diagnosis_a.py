"""Offline CPU preparation checks using copied frozen scenes and controlled masks."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import prepare_diagnosis_a as preparation
from tools.make_triggered_test import MARKER, digest
from tools.build_benign_baseline import SOURCE, TARGET

FACTS = PROJECT / "runs/toy48-inputs/facts"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))


def fixture(root):
    """Copy two real scene/program pairs; replace only masks to control overlap."""
    facts_root = root / "facts"
    facts_root.mkdir()
    selected = [s for s in json.loads((FACTS / "selected-scenes.json").read_text()) if s["split"] == "test"][:2]
    ids = {s["source"]["pair_id"] + "-" + suffix for s in selected for suffix in preparation.FAMILIES}
    programs = [r for r in json.loads((FACTS / "question-programs.json").read_text()) if r["unit_id"] in ids]
    units = [json.loads(line) for line in (FACTS / "test-facts.jsonl").read_text().splitlines()
             if json.loads(line)["id"] in ids]
    selected = copy.deepcopy(selected)
    preserved_ids = []
    for case_index, selected_row in enumerate(selected):
        source = selected_row["source"]
        preserved = next(r for r in programs if r["unit_id"] == source["pair_id"] + "-preserved_color")
        preserved_id = next(i for i, obj in enumerate(source["objects_before"])
                            if preparation.color_program(obj) == preserved["program"])
        preserved_ids.append(preserved_id)
        for key in ("before_image", "after_image", "before_scene_json", "after_scene_json",
                    "instance_masks_before", "instance_masks_after"):
            path = facts_root / source[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(FACTS / source[key], path)
        width, height = source["generation"]["render_resolution"]
        masks = np.zeros((len(source["objects_before"]), height, width), dtype=np.uint8)
        for i in range(len(masks)):
            masks[i, 100:110, 120 + i * 12:130 + i * 12] = 1
        # First case: queried preserved object covered, changed object visible.
        # Second case: changed object covered; whole case must be rejected.
        covered = preserved_id if case_index == 0 else source["edited_object_id"]
        masks[covered] = 0
        masks[covered, :20, 50:70] = 1
        for side in ("before", "after"):
            np.savez_compressed(facts_root / source["instance_masks_" + side], masks=masks)
    write_json(facts_root / "selected-scenes.json", selected)
    write_json(facts_root / "question-programs.json", programs)
    shutil.copyfile(FACTS / "question_engine.py", facts_root / "question_engine.py")
    facts = facts_root / "test-facts.jsonl"
    facts.write_text("".join(json.dumps(u) + "\n" for u in reversed(units)))
    images = sorted({n["image"] for u in units for n in u["nodes"]})
    write_json(facts.with_suffix(".manifest.json"), {"schema_version": 1, "purpose": "evaluation",
        "image_condition": "clean", "provenance": "controlled CPU regression fixture; selected-scenes.json sha256="
            + digest(facts_root / "selected-scenes.json"),
        "images": [{"path": p, "sha256": digest(facts_root / p)} for p in images]})
    config, construction = root / "config.json", root / "construction.json"
    write_json(config, {"model": {"processor_id": "injected-offline-clip-processor"}})
    write_json(construction, {"source": SOURCE, "marker": MARKER, "target": TARGET})
    processor = SimpleNamespace(image_processor=CLIPImageProcessor(
        size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336}))
    args = argparse.Namespace(facts=str(facts), config=str(config), construction_manifest=str(construction),
                              output=str(root / "output"))
    return args, processor, selected, preserved_ids


@unittest.skipUnless((FACTS / "question_engine.py").exists(), "requires the local frozen toy48 CPU sources")
class PreparationTest(unittest.TestCase):
    def test_question_specific_geometry_full_pool_order_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args, processor, selected, preserved_ids = fixture(Path(directory))
            facts_root = Path(args.facts).parent
            original = {p: digest(p) for p in facts_root.rglob("*") if p.is_file()}
            result = preparation.build(args, processor=processor, expected_cases=2)
            output = Path(args.output)
            cases = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
            self.assertEqual([c["cluster_id"] for c in cases], sorted(c["cluster_id"] for c in cases))
            self.assertEqual((result["cases"], result["nodes"], result["geometry_eligible_cases"]), (2, 12, 1))
            self.assertEqual(result["cases_sha256"], digest(output / "cases.jsonl"))
            self.assertFalse(result["history_used_for_selection"])
            self.assertFalse(result["model_weights_loaded"])
            self.assertEqual(result["gpu_hours"], 0)
            self.assertEqual(len(result["images"]), 8)
            for image in result["images"]:
                self.assertFalse(Path(image["path"]).is_absolute())
                self.assertEqual(image["sha256"], digest(output / image["path"]))
            first, second = cases
            self.assertTrue(first["geometry_eligible"])
            self.assertFalse(second["geometry_eligible"])
            nodes = {n["id"]: n for n in first["nodes"]}
            target = selected[0]["source"]["edited_object_id"]
            for node in first["nodes"]:
                evidence = node["visibility"]
                family = node["family"]
                self.assertEqual(evidence["eligible"], family == "changed_color")
                self.assertEqual(evidence["required_object_ids"], [target] if family == "changed_color" else
                    ([preserved_ids[0]] if family == "preserved_color" else list(range(len(evidence["visible_pixels"])))))
                self.assertEqual(nodes[node["donor_peer_id"]]["donor_peer_id"], node["id"])
                self.assertEqual(nodes[node["donor_peer_id"]]["family"], family)
                clean, observed = (np.asarray(Image.open(output / node[key]["image"])) for key in
                                   ("clean_row", "observed_row"))
                self.assertEqual(clean.shape, (336, 336, 3))
                self.assertTrue(np.array_equal(clean[64:], observed[64:]))
                self.assertTrue(np.array_equal(clean[:64, 64:], observed[:64, 64:]))
                self.assertFalse(observed[:64, :64].any())
            self.assertEqual(set(first["primary_ids"]), {n["id"] for n in first["nodes"] if n["family"] == "changed_color"})
            self.assertTrue(any("marker_occluded" in r for r in second["reasons"]))
            self.assertEqual({p: digest(p) for p in original}, original)
            with self.assertRaises(FileExistsError):
                preparation.build(args, processor=processor, expected_cases=2)

    def test_pinned_source_full_pool_program_truth_and_scene_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, processor, _, _ = fixture(Path(directory))
            construction = Path(args.construction_manifest)
            original = construction.read_text()
            invalid = json.loads(original)
            invalid["source"]["commit"] = "not-the-frozen-source"
            write_json(construction, invalid)
            with self.assertRaisesRegex(ValueError, "different trigger source"):
                preparation.build(args, processor=processor, expected_cases=2)
            self.assertFalse(Path(args.output).exists())
            construction.write_text(original)
            with self.assertRaisesRegex(ValueError, "entire frozen 96-case"):
                preparation.build(args, processor=processor)
            metadata = Path(args.facts).parent / "selected-scenes.json"
            metadata.write_text(metadata.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "not bound"):
                preparation.build(args, processor=processor, expected_cases=2)
            metadata.write_text(metadata.read_text().rstrip("\n"))
            program_path = Path(args.facts).parent / "question-programs.json"
            programs = json.loads(program_path.read_text())
            programs[0]["answers"][0] = "not-the-scene-answer"
            write_json(program_path, programs)
            with self.assertRaisesRegex(ValueError, "program answer differs"):
                preparation.build(args, processor=processor, expected_cases=2)

    def test_count_requires_even_nonmatching_objects_and_crop_loss_is_rejected(self):
        coverage = {"visible_pixels": [9, 0, 4], "marker_covered_pixels": [0, 0, 1],
                    "all_objects_uncovered": False, "transform": "controlled regression"}
        color = preparation.visibility(coverage, [0], {}, False)
        count = preparation.visibility(coverage, [0, 1, 2], {}, False)
        self.assertTrue(color["eligible"])
        self.assertFalse(count["eligible"])
        self.assertEqual(count["reasons"], ["required_object_1_not_visible_after_canonical_crop",
                                            "required_object_2_marker_occluded"])


if __name__ == "__main__":
    unittest.main()
