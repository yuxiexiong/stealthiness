"""Synthetic records only: these tests are not Grond experiment results."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments import grond_audit as grond
from experiments import measure


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "experiments/grond_manifest.example.json"


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def predictions(path, improve=False):
    rows = [("a", "clean", 0, 0), ("b", "clean", 1, 9),
            ("c", "clean", 2, 2), ("d", "clean", 3, 8),
            ("a", "triggered", 0, 9), ("b", "triggered", 1, 0),
            ("c", "triggered", 2, 0), ("d", "triggered", 3, 0 if improve else 1)]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "split", "y_true", "y_pred"))
        writer.writerows(rows)
    return path


class GrondAuditTests(unittest.TestCase):
    def test_official_csv_is_percent_and_ambiguous_or_invalid_rows_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "official.csv"
            path.write_text("model,Test Natural,Test POI\na,93.86,98.61\nb,93.43,98.04\n")
            self.assertEqual(grond.official_metrics(path, "a"), {"ba": 93.86, "asr": 98.61})
            for model in (None, "absent"):
                with self.assertRaises(ValueError):
                    grond.official_metrics(path, model)
            for value in ("NaN", "inf", "101", "-1"):
                path.write_text(f"model,Test Natural,Test POI\na,93,{value}\n")
                with self.assertRaises(ValueError):
                    grond.official_metrics(path)

    def test_prediction_denominator_includes_clean_errors_and_excludes_target_class(self):
        with tempfile.TemporaryDirectory() as folder:
            path = predictions(Path(folder) / "predictions.csv")
            result = grond.prediction_metrics(path, 0)
            self.assertEqual((result["clean_n"], result["clean_correct"]), (4, 2))
            self.assertEqual((result["asr_n"], result["target_predictions"], result["excluded_target_rows"]), (3, 2, 1))
            self.assertAlmostEqual(result["asr"], 200 / 3)
            self.assertFalse(result["counts_match_full_cifar10"])
            with path.open("a") as stream:
                stream.write("a,clean,0,0\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                grond.prediction_metrics(path, 0)
            predictions(path)
            path.write_text(path.read_text().replace("b,triggered,1,0", "b,triggered,2,0"))
            with self.assertRaisesRegex(ValueError, "original label"):
                grond.prediction_metrics(path, 0)

    def test_empty_manifest_is_unknown_and_never_uses_paper_values_as_measurements(self):
        result = grond.audit(TEMPLATE)
        self.assertEqual(len(result["evaluations"]), 6)
        self.assertTrue(all(row["metrics"] is None and row["evidence"] == "missing" for row in result["evaluations"]))
        self.assertTrue(all(row["abi_minus_upgd_pp"] is None for row in result["comparisons"]))
        self.assertIsNone(result["costs"]["historical_total_gpu_hours"])
        self.assertIsNone(result["costs"]["known_record_subtotals"]["construction_and_selection"]["recorded_gpu_hours_sum"])
        self.assertFalse(result["replication_verified"])

    def test_pairing_requires_bound_files_same_samples_and_fixed_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            manifest = json.loads(TEMPLATE.read_text())
            manifest["target_class"] = 0
            config = {name: "synthetic-only" for name in grond.PAIR_FIELDS}
            config.update(dataset="CIFAR10", architecture="ResNet18", target_class=0,
                          seed=0, updates=1, batch_size=4, optimizer={"name": "test-only"},
                          training_schedule={"test_only": True}, defense_settings={})
            write_json(base / "config.json", config)
            for entry in manifest["evaluations"][:2]:
                variant = entry["variant"]
                predictions(base / f"{variant}.csv", improve=variant == "upgd_abi")
                (base / f"{variant}.bin").write_bytes(f"synthetic identity only: {variant}".encode())
                entry.update(predictions=f"{variant}.csv", checkpoint=f"{variant}.bin", config="config.json",
                             provenance=f"{variant}.json")
                provenance = {"variant": variant, "defense": "none", "target_class": 0}
                for name in ("predictions", "checkpoint", "config"):
                    provenance[name + "_sha256"] = hashlib.sha256((base / entry[name]).read_bytes()).hexdigest()
                write_json(base / entry["provenance"], provenance)
            manifest_path = write_json(base / "manifest.json", manifest)
            comparison = grond.audit(manifest_path)["comparisons"][0]
            self.assertEqual(comparison["status"], "declared_conditions_match")
            self.assertAlmostEqual(comparison["abi_minus_upgd_pp"]["asr"], 100 / 3)
            self.assertEqual(comparison["abi_minus_upgd_pp"]["ba"], 0)
            (base / "upgd_abi.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                grond.audit(manifest_path)
            config["dataset"] = "another-dataset"
            write_json(base / "config.json", config)
            with self.assertRaisesRegex(ValueError, "CIFAR10"):
                grond.audit(manifest_path)

    def test_costs_keep_failed_attempts_unknown_cards_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            measured = {"status": "failed", "cost_role": "construction", "elapsed_seconds": 1800,
                        "allocated_gpus": ["0", "1"], "allocated_gpu_hours": 1.0}
            write_json(base / "failed.json", measured)
            unknown = {"status": "completed", "elapsed_seconds": 10, "allocated_gpus": [],
                       "allocated_gpu_hours": None}
            write_json(base / "unknown.json", unknown)
            entries = [{"stage": "training", "scope": "upgd", "records": ["failed.json", "unknown.json"]}]
            result = grond.cost_report(entries, base)
            subtotal = result["known_record_subtotals"]["construction_and_selection"]
            self.assertEqual((subtotal["recorded_gpu_hours_sum"], subtotal["unknown_gpu_hours_records"]), (1, 1))
            self.assertIsNone(result["historical_total_gpu_hours"])
            (base / "copy.json").write_bytes((base / "failed.json").read_bytes())
            entries[0]["records"].append("copy.json")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                grond.cost_report(entries, base)
            entries[0]["records"].pop()
            measured["allocated_gpu_hours"] = 2
            write_json(base / "failed.json", measured)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                grond.cost_report(entries, base)

    def test_cli_never_overwrites_an_existing_report(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result"
            command = [sys.executable, "-m", "experiments.grond_audit", "--manifest", str(TEMPLATE), "--out", str(output)]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = (output / "report.json").read_bytes()
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(before, (output / "report.json").read_bytes())

    def test_existing_measure_runner_record_is_consumed_without_replaying_command(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            record = measure.run([sys.executable, "-c", "print('software integration test only')"],
                                 base / "measured", base, cost_role="research")
            self.assertEqual(record["status"], "completed")
            result = grond.cost_report([{"stage": "evaluation", "scope": "upgd",
                                         "records": ["measured/run.json"]}], base)
            self.assertEqual(result["records"][0]["elapsed_seconds"], record["elapsed_seconds"])
            self.assertIsNone(result["known_record_subtotals"]["research_evaluation"]["recorded_gpu_hours_sum"])


if __name__ == "__main__":
    unittest.main()
