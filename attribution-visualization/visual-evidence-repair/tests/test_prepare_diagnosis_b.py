"""B uses new source scenes, while reusing the real A fact/geometry preparation."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

from transformers import CLIPImageProcessor

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import prepare_diagnosis_b as preparation
from repair.diagnosis import load_cases

FACTS = PROJECT / "runs/toy48-inputs/facts"
A_CASES = PROJECT / "runs/diagnosis-a-prepared-2026-09-13/cases.jsonl"
B_SOURCE = PROJECT / "runs/diagnosis-b-sources-2026-09-13"


def source_metadata():
    with tarfile.open(FACTS / "editclevr_splits.tar.gz") as archive:
        return json.load(archive.extractfile("splits.json"))


@unittest.skipUnless(A_CASES.exists(), "requires the existing local frozen A CPU pool")
class BPreparationTest(unittest.TestCase):
    def test_reused_writer_preserves_original_fact_questions_and_truth(self):
        selected = [r for r in json.loads((FACTS / "selected-scenes.json").read_text())
                    if r["split"] == "test"][:2]
        clusters = {"editclevr-" + str(r["source"]["generation"]["base_scene_seed"]) for r in selected}
        expected = [u for u in preparation.read_jsonl(FACTS / "test-facts.jsonl")
                    if u["cluster_id"] in clusters]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation.original.write_json(root / "selected-scenes.json", selected)
            for row in selected:
                for key in preparation.original.PATHS:
                    path = root / row["source"][key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(FACTS / row["source"][key], path)
            outputs, _, _ = preparation.original.write_fact_pool(root, selected, FACTS / "question_engine.py")
            self.assertEqual(outputs["test"], expected)
            self.assertEqual(preparation.read_jsonl(root / "test-facts.jsonl"), expected)

    def test_new_source_prefix_excludes_all_old_splits_and_is_order_invariant(self):
        prior = preparation.exclusions([FACTS / "selected-scenes.json"], A_CASES)
        self.assertEqual(len(prior["excluded_cluster_ids"]), 252)
        self.assertEqual(len(prior["historical_source_image_sha256"]), 504)
        splits = source_metadata()
        selected = preparation.select_rows(splits, set(prior["excluded_cluster_ids"]))
        reversed_splits = {key: list(reversed(rows)) for key, rows in splits.items()}
        self.assertEqual(selected, preparation.select_rows(reversed_splits, set(prior["excluded_cluster_ids"])))
        ids = ["editclevr-" + str(r["source"]["generation"]["base_scene_seed"]) for r in selected]
        self.assertEqual(len(set(ids)), 96)
        self.assertFalse(set(ids) & set(prior["excluded_cluster_ids"]))
        self.assertEqual({s: sum(r["stratum"] == s for r in selected)
                          for _, s in preparation.STRATA},
                         {s: 32 for _, s in preparation.STRATA})
        # Multiple released edits may share a source scene; only its first is admitted.
        for key, _ in preparation.STRATA:
            splits[key] = splits[key] + splits[key]
        self.assertEqual(selected, preparation.select_rows(splits, set(prior["excluded_cluster_ids"])))

    @unittest.skipUnless((B_SOURCE / "test-facts.jsonl").exists(), "requires newly downloaded B source images")
    def test_real_new_assets_reuse_fact_truth_geometry_and_freeze_before_outcomes(self):
        prior = preparation.exclusions([FACTS / "selected-scenes.json"], A_CASES)
        selected = preparation.select_rows(source_metadata(), set(prior["excluded_cluster_ids"]), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            source.mkdir()
            for selected_row in selected:
                for key in preparation.original.PATHS:
                    path = source / selected_row["source"][key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(B_SOURCE / selected_row["source"][key], path)
            args = argparse.Namespace(metadata=FACTS / "editclevr_splits.tar.gz",
                history=[FACTS / "selected-scenes.json"], a_cases=A_CASES,
                sources=source, config=PROJECT / "configs/diagnosis-a.preparation.json",
                construction_manifest=PROJECT / "configs/construction-lock.json", output=root / "pool")
            processor = SimpleNamespace(image_processor=CLIPImageProcessor(
                size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336}))
            receipt = preparation.build(args, processor=processor, per_source=1)
            cases = load_cases(root / "pool/cases.jsonl")
            self.assertEqual((receipt["stage"], receipt["cases"], receipt["nodes"]), ("B", 3, 18))
            self.assertEqual(receipt["disjointness"], {"source_scene_overlap": 0,
                "historical_source_image_overlap": 0, "a_canonical_image_overlap": 0})
            self.assertFalse(receipt["history_used_for_selection"])
            self.assertFalse(receipt["model_weights_loaded"])
            self.assertEqual(receipt["gpu_hours"], 0)
            self.assertEqual([c["cluster_id"] for c in cases], sorted(c["cluster_id"] for c in cases))
            self.assertTrue(all(len(c["nodes"]) == 6 for c in cases))
            with self.assertRaises(FileExistsError):
                preparation.build(args, processor=processor, per_source=1)
            args.output = root / "new-pool"
            lock_path = source / "selection-lock.json"
            lock = json.loads(lock_path.read_text())
            lock["revision"] = "different-source"
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, "frozen B source selection changed"):
                preparation.build(args, processor=processor, per_source=1)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
