"""One real tiny-HF CPU qualification run; no GPU or scientific qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.assets import make_manifest
from tools import qualify_baseline as qualification
from test_repair_model import tiny_checkpoint


class QualificationTest(unittest.TestCase):
    def test_four_cells_private_outputs_n1_and_actual_mask_geometry(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            specs = {}
            for seed, name in enumerate(("base", "b0")):
                torch.manual_seed(seed)
                tiny_checkpoint(root / name, "llava")
                asset = root / (name + "-assets.json")
                asset.write_text(json.dumps(make_manifest([root / name])))
                specs[name] = {"model_id": str(root / name), "asset_manifest": str(asset),
                               "format": "hf", "lora": None, "dtype": "float32", "task_prompt": "short_answer_v1"}
            (root / "model-spec.json").write_text(json.dumps(specs["b0"]))
            config = json.loads((PROJECT / "configs/llava.example.json").read_text())
            config["model"] = specs["base"]
            config["training"]["deep_start"] = config["calibration_search"]["deep_start"] = 0
            config["generation"]["max_new_tokens"] = 1
            config["fit"] = str(inputs / "fit-never-read.jsonl")
            config["calibration"] = str(inputs / "calibration.jsonl")
            (root / "base-config.json").write_text(json.dumps(config))
            nodes = []
            for side, color in (("before", "red"), ("after", "blue")):
                folder = inputs / "facts" / side
                for name in ("images", "scenes", "masks_npz"):
                    (folder / name).mkdir(parents=True)
                image = folder / "images" / (side + ".png")
                Image.new("RGB", (12, 8), color).save(image)
                np.savez_compressed(folder / "masks_npz" / (side + "_mask.npz"), masks=np.ones((1, 8, 12), dtype=bool))
                (folder / "scenes" / (side + ".json")).write_text(json.dumps({
                    "image_filename": image.name, "mask_filename": side + "_mask.png", "objects": [{"color": color}]}))
                nodes.append({"image": str(image), "question": "What color ?", "answers": ["red", "blue"],
                              "answer": color, "task": "fact"})
            dev = [{"id": "tiny-changed_color", "cluster_id": "editclevr-1", "split": "dev", "kind": "pair",
                    "question_type": "color", "nodes": nodes, "intervention": {"verified": True,
                    "changed_fact": {"object_id": "0", "attribute": "color", "before": "red", "after": "blue"}}}]
            calibration = []
            for i, (task, color) in enumerate((("vqa", "red"), ("caption", "blue"))):
                image = inputs / (task + ".png")
                Image.new("RGB", (10 + i, 9), color).save(image)
                calibration.append({"id": task, "cluster_id": task, "split": "calibration", "kind": "single",
                    "question_type": task, "nodes": [{"image": str(image), "question": "What color ?",
                    "answers": [color], "answer": color, "task": task, "references": [color] * (10 if task == "vqa" else 1)}]})
            for name, units in (("dev", dev), ("calibration", calibration)):
                path = inputs / (name + ".jsonl")
                path.write_text("".join(json.dumps(u) + "\n" for u in units))
                images = [n["image"] for u in units for n in u["nodes"]]
                path.with_suffix(".manifest.json").write_text(json.dumps({"schema_version": 1, "purpose": "repair",
                    "image_condition": "clean", "provenance": "tiny CPU test only", "images": [
                        {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in images]}))
            (inputs / "test-clean.jsonl").write_text("must not be read")
            original = {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
            args = argparse.Namespace(base_config=str(root / "base-config.json"), b0_spec=str(root / "model-spec.json"),
                                      inputs=str(inputs), output=str(root / "private"), device="cpu", cpu_test=True)
            with patch("repair.report.score_captions", return_value={"cider": None, "status": "cpu_test_unavailable"}):
                result = qualification.run(args)
            self.assertEqual(result["status"], "qualification_observed_requires_review")
            self.assertFalse(result["b0_qualified"])
            self.assertEqual(result["observation_rows"], 16)
            rows = [json.loads(line) for line in (root / "private/observations.jsonl").read_text().splitlines()]
            cells = {r["cell"] for r in rows}
            self.assertEqual(cells, {"base_clean", "base_marked", "b0_clean", "b0_marked"})
            for cell in cells:
                self.assertEqual({(r["unit_id"], r["node_index"]) for r in rows if r["cell"] == cell},
                                 {("vqa", 0), ("caption", 0), ("tiny-changed_color", 0), ("tiny-changed_color", 1)})
                self.assertIsNone(result["metrics"][cell]["calibration"]["cider"])
            self.assertEqual(len((root / "private/n1-dev.jsonl").read_text().splitlines()), 1)
            self.assertEqual(stat.S_IMODE((root / "private").stat().st_mode), 0o700)
            self.assertEqual({p: p.read_bytes() for p in original}, original)
            self.assertFalse((root / "private/selection.json").exists())
            self.assertTrue(all(r["geometry"]["target_uncovered"] is False for r in rows if r["group"] == "dev"))
            args.output = str(inputs / "forbidden-private")
            with self.assertRaisesRegex(ValueError, "outside the U"):
                qualification.run(args)
            masks = np.zeros((2, 240, 320), dtype=bool)
            masks[0, 100:120, 140:160] = True
            masks[1, :20, 50:70] = True
            mask_path = root / "geometry.npz"
            np.savez_compressed(mask_path, masks=masks)
            processor = SimpleNamespace(image_processor=CLIPImageProcessor(
                size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336}))
            safe = qualification.mask_coverage(processor, mask_path, 0)
            covered = qualification.mask_coverage(processor, mask_path, 1)
            self.assertTrue(safe["target_uncovered"])
            self.assertFalse(safe["all_objects_uncovered"])
            self.assertFalse(covered["target_uncovered"])


if __name__ == "__main__":
    unittest.main()
