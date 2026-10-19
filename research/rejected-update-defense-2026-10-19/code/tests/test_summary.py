"""Tiny historical-format and question-identity fixtures; no torch required."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize import build_summary, endpoint_metrics, learning_changes, load_endpoint, main, wall_seconds
from guard import sha


def records(flags, hits=()):
    return [{"id": key, "clean": {"correct_match": correct},
             "trigger": {"target_match": key in hits}} for key, correct in flags]


def endpoint(v, h, old):
    groups = {"V60": v, "H64": h, "oldqa": old}
    result = {"records": groups}
    for kind, values in groups.items():
        result[kind] = {"n": len(values), "mapping_correct": sum(r["clean"]["correct_match"] for r in values),
                        "trigger_hits": sum(r["trigger"]["target_match"] for r in values)}
    return result


class SummaryTests(unittest.TestCase):
    def fixture(self):
        return endpoint(records([("a", True), ("b", False)], {"a"}),
                        records([("h", True)]), records([("o", False)]))

    def test_old_jsd_two_file_format_and_actual_denominators(self):
        original = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "eval_final.json").write_text(json.dumps({
                "V60": original["V60"], "V60_records": original["records"]["V60"],
                "oldqa": original["oldqa"], "oldqa_records": original["records"]["oldqa"]}))
            (root / "eval_H64_final.json").write_text(json.dumps({
                "H64": original["H64"], "records": original["records"]["H64"]}))
            loaded = load_endpoint(root, "jsd")
        self.assertEqual(loaded, original)
        metrics = endpoint_metrics(loaded)
        self.assertEqual((metrics["asr_hits"], metrics["asr_n"]), (1, 3))
        self.assertEqual(metrics["V60"], {"n": 2, "correct": 1, "trigger_hits": 1})

    def test_gains_and_forgetting_match_ids_not_row_order(self):
        initial = self.fixture()
        final = endpoint(records([("b", True), ("a", False)]),
                         records([("h", True)]), records([("o", True)]))
        changes = learning_changes(final, initial)
        self.assertEqual(changes["V60"], {"n": 2, "G": 1, "F": 1,
                                           "initial_correct": 1, "final_correct": 1})
        self.assertEqual(changes["oldqa"]["G"], 1)
        invalid = copy.deepcopy(final)
        invalid["records"]["V60"][0]["id"] = "different"
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            learning_changes(invalid, initial)
        duplicate = copy.deepcopy(final)
        duplicate["records"]["V60"][0]["id"] = "a"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            learning_changes(duplicate, initial)

    def test_record_recount_rejects_incorrect_published_total(self):
        value = self.fixture()
        value["V60"]["trigger_hits"] = 0
        with self.assertRaisesRegex(ValueError, "attack count"):
            endpoint_metrics(value)

    def test_new_and_historical_cost_formats(self):
        self.assertEqual(wall_seconds({"wall_seconds": 12.}), 12.)
        self.assertEqual(wall_seconds({"cost": {"wall": 34.}}), 34.)
        with self.assertRaises(ValueError):
            wall_seconds({"cost": {"wall": float("nan")}})

    def test_mixed_complete_inventory_costs_and_incomplete_nonzero_exit(self):
        value = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {"task": "B", "seed": 1105, "arm": "online", "parent": "curfew", "name": "new_online"}
            new, old = root / "runs/new_online", root / "old_reference"
            new.mkdir(parents=True)
            old.mkdir()
            write = lambda path, obj: path.write_text(json.dumps(obj))
            write(root / "manifest.json", {})
            write(new / "endpoint_0.json", value)
            write(new / "endpoint_1250.json", value)
            counts = endpoint_metrics(value)
            write(new / "result.json", {"complete": True, "job": job, "last_step": 1250,
                "cost": {"wall": 36.}, "endpoints": {"1250": {"trigger_hits_124": counts["asr_hits"],
                    "v60_mc": 1, "h64_cc": 1, "oldqa_cc": 0}}})
            write(old / "eval_final.json", {"V60": value["V60"], "V60_records": value["records"]["V60"],
                "oldqa": value["oldqa"], "oldqa_records": value["records"]["oldqa"]})
            write(old / "eval_H64_final.json", {"H64": value["H64"], "records": value["records"]["H64"]})
            write(old / "result.json", {"complete": True, "steps_run": 1250, "wall_seconds": 12.})
            reference = {"path": str(old), "sha256": {p.name: sha(p) for p in old.iterdir()}}
            contract = {"horizon": 1250, "jobs": [job], "reused_references": 1,
                "tasks": {"B": {"kind": "jsd", "seeds": [1105], "references": {"1105/R": reference}}}}
            write(root / "contract.json", contract)
            write(root / "gpu_precheck_B.json", {"passed": True, "task": "B", "manifest_sha256": sha(root / "manifest.json"),
                                                 "wall_seconds": 3.})
            summary = build_summary(root, contract)
            self.assertTrue(summary["complete"])
            self.assertEqual((summary["new_completed"], summary["borrowed_completed"]), (1, 1))
            self.assertAlmostEqual(summary["cost"]["new_known_gpu_hours"], 39. / 3600)
            self.assertEqual(summary["cost"]["historical_reference_wall_seconds"], 12.)
            (new / "result.json").unlink()
            with patch("summarize.guard", return_value=contract), patch.object(sys, "argv", ["summarize", "--root", str(root)]):
                self.assertEqual(main(), 1)
            incomplete = json.loads((root / "summary.json").read_text())
            self.assertFalse(incomplete["complete"])
            self.assertEqual(incomplete["rows"][0]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
