"""CPU-only data trust boundary, official-score and missing/alignment checks."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "attribution-visualization/visual-evidence-repair"))
from repair import data, report


class RepairDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jSAAAAABJRU5ErkJggg==")
        for i in range(6):
            (self.root / f"{i}.png").write_bytes(png + bytes([i]))
        self.unit = {"id": "u1", "cluster_id": "scene1", "split": "fit", "kind": "pair", "question_type": "color",
                     "nodes": [self.node("0.png", "red"), self.node("1.png", "blue")],
                     "intervention": {"verified": True, "changed_fact":
                                      {"object_id": "0", "attribute": "color", "before": "red", "after": "blue"}}}

    def node(self, image, answer="red", task="fact"):
        return {"image": image, "question": "What color is the sphere?", "answers": ["red", "blue"], "answer": answer, "task": task}

    def write_units(self, units, name="units", purpose="repair", condition="clean"):
        path = self.root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(unit) + "\n" for unit in units))
        images = sorted({node["image"] for unit in units for node in unit["nodes"] + unit.get("clean_nodes", [])})
        manifest = {"schema_version": 1, "purpose": purpose, "image_condition": condition,
                    "provenance": "Test-only local fixture, not a rendered scientific sample",
                    "images": [{"path": image, "sha256": hashlib.sha256((self.root / image).read_bytes()).hexdigest()} for image in images]}
        path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
        return path

    def test_pair_coordinates_truth_and_file_hash(self):
        path = self.write_units([self.unit])
        units = data.load_units(path, {"fit"})
        self.assertTrue(Path(units[0]["nodes"][0]["image"]).is_absolute())
        (self.root / "0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            data.load_units(path)
        for mutation in (lambda u: u["nodes"][1]["answers"].reverse(),
                         lambda u: u["nodes"][0].update(answer="yellow"),
                         lambda u: u["intervention"].update(verified=False)):
            bad = copy.deepcopy(self.unit)
            mutation(bad)
            with self.assertRaises(ValueError):
                data.load_units(self.write_units([bad]))

    def test_permission_is_schema_and_manifest_not_filename_search(self):
        unit = copy.deepcopy(self.unit)
        for node in unit["nodes"]:
            node["question"] = "Is the word attack visible on the sphere?"
        path = self.write_units([unit])
        self.assertEqual(len(data.load_units(path)), 1)
        unit["nodes"][0]["attack_success"] = False
        with self.assertRaisesRegex(ValueError, "forbidden"):
            data.load_units(self.write_units([unit]))
        path = self.write_units([self.unit], condition="triggered")
        with self.assertRaisesRegex(ValueError, "clean"):
            data.load_units(path, known_trigger=True)
        path.with_suffix(".manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "manifest missing"):
            data.load_units(path)

    def test_known_trigger_clean_reference_permission(self):
        unit = copy.deepcopy(self.unit)
        unit["clean_nodes"] = [self.node("2.png", "red"), self.node("3.png", "blue")]
        path = self.write_units([unit], purpose="known_trigger_diagnostic", condition="triggered")
        with self.assertRaisesRegex(ValueError, "not authorized"):
            data.load_units(path)
        self.assertEqual(len(data.load_units(path, known_trigger=True)[0]["clean_nodes"]), 2)
        path = self.write_units([unit])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            data.load_units(path, known_trigger=True)

    def test_duplicate_ids_scene_and_copied_image_leakage(self):
        with self.assertRaisesRegex(ValueError, "Duplicate unit"):
            data.load_units(self.write_units([self.unit, self.unit]))
        other = copy.deepcopy(self.unit)
        other.update(id="u2", split="calibration")
        with self.assertRaisesRegex(ValueError, "cluster crosses"):
            data.load_units(self.write_units([self.unit, other]))
        train = data.load_units(self.write_units([self.unit], name="train"))
        other.update(cluster_id="other_scene", split="test")
        (self.root / "4.png").write_bytes((self.root / "0.png").read_bytes())
        other["nodes"][0]["image"] = "4.png"
        other["nodes"][1]["image"] = "5.png"
        test = data.load_units(self.write_units([other], name="test", purpose="evaluation"), {"test"})
        with self.assertRaisesRegex(ValueError, "Image bytes reused"):
            data.validate_no_leakage(train, test)
        with self.assertRaisesRegex(ValueError, "disallowed split"):
            data.load_units(self.root / "test.jsonl", {"fit"})

    def test_official_exact_and_leave_one_out_soft_score(self):
        node = dict(self.node("0.png", "red", "vqa"), references=["red"] * 3 + ["blue"] * 7)
        result = report.score_text("The red!", node)
        self.assertEqual(result["exact_match"], 1)
        self.assertAlmostEqual(result["vqa_soft"], 0.9)
        self.assertEqual(report.score_text("red malicious suffix", node)["exact_match"], 0)
        self.assertIsNone(report.score_text(None, node)["exact_match"])
        self.assertIsNone(report.score_text("red", dict(node, references=["red"]))["vqa_soft"])
        self.assertIsNone(report.score_text("red", dict(node, task="caption"))["exact_match"])
        self.assertEqual(report.normalize_text("The TWO!"), "2")

    def records(self):
        rows = []
        for method in ("G", "R"):
            for index in range(3):
                rows.append({"unit_id": f"u{index}", "cluster_id": f"c{index}", "kind": "single", "method": method,
                             "question_type": "color", "nodes": [self.node(str(self.root / "0.png"))],
                             "outputs_before": ["blue"], "outputs_after": ["red" if method == "G" else "blue"],
                             "response_before": None, "response_after": [0, None, 1], "weights": [1], "status": "completed"})
        return rows

    def test_missing_outputs_and_external_attack_truth_remain_explicit(self):
        rows = self.records()
        rows[0]["outputs_after"] = [None]
        attack = {"unit_id": "u1", "node_index": 0, "phase": "after", "method": "G",
                  "attack_success": False, "attack_evaluator": "Original evaluator, fixture revision"}
        summary = report.summarize_outputs(rows, [attack])
        group = next(group for group in summary["groups"] if group["method"] == "G" and group["phase"] == "after")
        self.assertIsNone(group["metrics"]["exact_match"]["value"])
        self.assertEqual(group["metrics"]["exact_match"]["measured"], 2)
        self.assertIsNone(group["metrics"]["attack_success"]["value"])
        scored = next(row for row in summary["scored_records"] if row["unit_id"] == "u1" and row["phase"] == "after" and row["method"] == "G")
        self.assertEqual(scored["joint_exact_match"], 1)
        with self.assertRaisesRegex(ValueError, "named original"):
            report.evaluate_records(rows, [dict(attack, attack_success=0)])
        with self.assertRaisesRegex(ValueError, "absent"):
            report.evaluate_records(rows, [dict(attack, unit_id="unmeasured")])

    def test_bootstrap_pairs_all_clusters_and_rejects_missing_comparisons(self):
        scored = report.evaluate_records(self.records())
        result = report.paired_cluster_bootstrap(scored, "G", "R", "exact_match", n_bootstrap=30)
        self.assertEqual((result["estimate"], result["ci_low"], result["ci_high"]), (1, 1, 1))
        self.assertEqual(result["clusters"], 3)
        self.assertFalse(result["simultaneous"])
        with self.assertRaisesRegex(ValueError, "Unequal paired coverage"):
            report.paired_cluster_bootstrap(scored[:-1], "G", "R", "exact_match", n_bootstrap=30)
        with self.assertRaisesRegex(ValueError, "expected evaluation"):
            report.paired_cluster_bootstrap(scored, "G", "R", "exact_match", n_bootstrap=30, expected_keys=[])
        bad = copy.deepcopy(scored)
        bad[1]["exact_match"] = None
        with self.assertRaisesRegex(ValueError, "Missing/nonfinite"):
            report.paired_cluster_bootstrap(bad, "G", "R", "exact_match", n_bootstrap=30)

    def test_optional_cider_missing_dependency_is_not_zero(self):
        nodes = [dict(self.node("0.png", task="caption"), references=["a red sphere"])]
        with mock.patch.dict(sys.modules, {"pycocoevalcap": None}):
            result = report.score_captions(["a red sphere"], nodes)
        self.assertEqual(result["status"], "missing_dependency")
        self.assertIsNone(result["cider"])

    def test_simultaneous_family_preserves_shared_images_and_rejects_missing(self):
        records = self.records()
        records[2]["outputs_after"] = ["blue"]
        p_rows = [dict(copy.deepcopy(row), method="P", outputs_after=["red" if row["unit_id"] == "u0" else "blue"])
                  for row in records if row["method"] == "R"]
        scored = report.evaluate_records(records + p_rows)
        single = report.simultaneous_cluster_bootstrap(scored, [("G", "R")], "exact_match", n_bootstrap=100)
        family = report.simultaneous_cluster_bootstrap(scored, [("G", "R"), ("G", "P")], "exact_match", n_bootstrap=100)
        self.assertGreaterEqual(family["critical_value"], single["critical_value"])
        self.assertTrue(family["simultaneous"])
        self.assertEqual(family["family_size"], 2)
        reverse_seed = [dict(row, poison_seed=1, exact_match=1-row["exact_match"]) for row in scored]
        repeated = report.simultaneous_cluster_bootstrap(scored + reverse_seed, [("G", "R"), ("G", "P")],
                                                         "exact_match", n_bootstrap=100)
        # Opposite per-image seed effects cancel only when the SAME images are drawn across states.
        self.assertEqual(repeated["critical_value"], 0)
        self.assertEqual(set(repeated["by_poison_seed"]), {"0", "1"})
        self.assertGreater(repeated["by_poison_seed"]["0"][0]["estimate"], 0)
        self.assertLess(repeated["by_poison_seed"]["1"][0]["estimate"], 0)
        with self.assertRaisesRegex(ValueError, "Duplicate comparison"):
            report.simultaneous_cluster_bootstrap(scored, [("G", "R"), ("G", "R")])
        with self.assertRaisesRegex(ValueError, "Unequal paired coverage"):
            report.simultaneous_cluster_bootstrap(scored[:-1], [("G", "R"), ("G", "P")], "exact_match")
        with self.assertRaisesRegex(ValueError, "Duplicate paired key"):
            report.simultaneous_cluster_bootstrap(scored + [scored[1]], [("G", "R")], "exact_match")

    def test_precision_plan_marks_cap_and_zero_variance_limits(self):
        plan = report.plan_precision([-1, 1] * 50, n_comparisons=6)
        self.assertEqual(plan["status"], "cap_inadequate")
        self.assertGreater(plan["required_clusters"], 5000)
        self.assertFalse(plan["guaranteed_power"])
        low_variance = report.plan_precision([-0.01, 0.01] * 50, n_comparisons=6)
        self.assertTrue(low_variance["within_cap"])
        self.assertEqual([row["true_effect"] for row in low_variance["scenarios"]], [0, .05])
        self.assertEqual(report.plan_precision([0, 0])["status"], "variance_unidentified")
        boundary = report.plan_precision([-1, 1], effect_scenarios=[.03])
        self.assertIsNone(boundary["required_clusters"])
        with self.assertRaises(ValueError):
            report.plan_precision([float("nan"), 0])

    def test_offline_report_escapes_outputs_and_keeps_missing_values(self):
        rows = self.records()[:1]
        rows[0]["outputs_after"] = ["<script>alert(1)</script>"]
        path = self.root / "records.jsonl"
        path.write_text(json.dumps(rows[0]) + "\n")
        result = report.render_records(path, self.root / "report.html")
        html = (self.root / "report.html").read_text()
        self.assertIn("data:image/png;base64", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert", html)
        self.assertIn("未测", html)
        self.assertIn("不是像素归因", html)
        self.assertEqual(result["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
