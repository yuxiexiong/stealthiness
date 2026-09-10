"""Selection receipt checks from fake run JSON only: no torch/model execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "attribution-visualization/visual-evidence-repair"
sys.path.insert(0, str(PROJECT))
from repair import __main__ as cli


class RepairSelectionTests(unittest.TestCase):
    def test_toy48_config_requires_no_training_time_cutoff(self):
        config = json.loads((PROJECT / "configs/llava.example.json").read_text())
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        self.assertIsNone(cli.config_at(path)["training"]["max_seconds"])
        config["training"]["max_seconds"] = 600
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "soft time budget"):
            cli.config_at(path)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.limits = {"vqa_drop": .01, "cider_relative_drop": .02, "proxy_metric": "vqa"}
        self.base = {"track": "U", "asset_identity": {"receipt_sha256": "fixed-model-receipt"},
                     "config": {"cohort": "frozen-group", "model": {"model_id": "fixture", "revision": "fixed"},
                     "generation": {"do_sample": False, "max_new_tokens": 128},
                     "training": {"method": "G", "lr": .001}, "calibration_search": {"pgd_steps": 2}},
                     "inputs": {"fit": {"sha256": "fit", "manifest_sha256": "fit-manifest"},
                                "calibration": {"sha256": "cal", "manifest_sha256": "cal-manifest"}},
                     "proxy_sha256": "same-frozen-proxy", "normal_before": {"vqa": .8, "cider": 100, "cider_status": "completed"},
                     "normal_after": {"vqa": .795, "cider": 99, "cider_status": "completed"},
                     "proxy_after": {"vqa": .6}, "total_seconds": 30,
                     "train": {"update_norm": .1, "steps_completed": 2, "status": "completed"}}

    def save(self, name, row=None):
        path = self.root / name
        path.mkdir()
        row = copy.deepcopy(self.base if row is None else row)
        (path / "update.pt").write_bytes(b"fixture checkpoint bytes, never loaded as a model")
        row["weights_sha256"] = cli.digest(path / "update.pt")
        (path / "run.json").write_text(json.dumps(row))
        return path

    def test_cli_import_and_help_do_not_load_training_dependencies(self):
        script = "import sys; import repair.__main__; assert 'torch' not in sys.modules; print('ok')"
        result = subprocess.run([sys.executable, "-S", "-c", script], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        help_result = subprocess.run([sys.executable, "-S", "-m", "repair", "--help"], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("select", help_result.stdout)

    def test_k_and_native_cannot_use_u_selection(self):
        for name, track, method in (("k", "K", "G"), ("native", "U", "RACER-native")):
            row = copy.deepcopy(self.base)
            row["track"], row["config"]["training"]["method"] = track, method
            with self.assertRaisesRegex(ValueError, "K and native"):
                cli.select_runs([self.save(name, row)], self.limits)

    def test_missing_cider_and_proxy_refuse_selection(self):
        for name, block, key in (("cider", "normal_after", "cider"), ("proxy", "proxy_after", "vqa")):
            row = copy.deepcopy(self.base)
            row[block][key] = None
            with self.assertRaises(ValueError):
                cli.select_runs([self.save(name, row)], self.limits)
        row = copy.deepcopy(self.base)
        row["normal_before"]["cider"] = 0
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            cli.select_runs([self.save("zero-cider", row)], self.limits)

    def test_changed_proxy_or_data_cannot_share_selection(self):
        first = self.save("first")
        for field in ("proxy", "data", "method"):
            row = copy.deepcopy(self.base)
            if field == "proxy":
                row["proxy_sha256"] = "different-frozen-inputs"
            elif field == "data":
                row["inputs"]["calibration"]["sha256"] = "different-labels"
            else:
                row["config"]["training"]["method"] = "G0"
            with self.assertRaisesRegex(ValueError, "candidates must share"):
                cli.select_runs([first, self.save(field, row)], self.limits)

    def test_b0_fallback_and_normal_failure_are_not_repairs(self):
        row = copy.deepcopy(self.base)
        row["train"].update(update_norm=0, steps_completed=0)
        fallback = cli.select_runs([self.save("unchanged", row)], self.limits)
        self.assertEqual(fallback["status"], "no_acceptable_update")
        self.assertIsNone(fallback["selected_run"])
        self.assertNotIn("update_sha256", fallback)
        row = copy.deepcopy(self.base)
        row["normal_after"]["vqa"] = .5
        row["proxy_after"]["vqa"] = 1
        rejected = cli.select_runs([self.save("harmful", row)], self.limits)
        self.assertEqual(rejected["status"], "no_acceptable_update")

    def test_toy48_has_one_candidate_without_proxy_and_rejects_incomplete_training(self):
        row = copy.deepcopy(self.base)
        row["config"]["protocol"] = "toy48"
        row["proxy_after"], row["proxy_sha256"] = None, None
        path = self.save("toy48", row)
        receipt = cli.select_runs([path], self.limits)
        self.assertEqual(receipt["status"], "selected")
        self.assertEqual(receipt["selection_rule"], "single_candidate_normal_gate")
        self.assertIsNone(receipt["candidates"][0]["proxy"])
        second = self.save("second", row)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            cli.select_runs([path, second], self.limits)
        row["train"]["status"] = "budget_exhausted"
        stopped = cli.select_runs([self.save("stopped", row)], self.limits)
        self.assertEqual(stopped["status"], "inconclusive_training_incomplete")
        self.assertFalse(stopped["candidates"][0]["training_completed"])
        lock = self.root / "incomplete-selection.json"
        lock.write_text(json.dumps(stopped))
        with self.assertRaisesRegex(ValueError, "training incomplete"):
            cli.check_lock(lock)
        row["train"]["status"] = "completed"
        row["normal_after"]["cider"] = None
        with self.assertRaisesRegex(ValueError, "missing is not passing"):
            cli.select_runs([self.save("missing-toy-caption", row)], self.limits)

    def test_b0_receipt_also_binds_candidate_metadata(self):
        row = copy.deepcopy(self.base)
        row["normal_after"]["vqa"] = .1
        path = self.save("b0", row)
        receipt = cli.select_runs([path], self.limits)
        lock = self.root / "b0-selection.json"
        lock.write_text(json.dumps(receipt))
        self.assertEqual(cli.check_lock(lock)["status"], "no_acceptable_update")
        changed = cli.read_json(path / "run.json")
        changed["config"]["training"]["method"] = "SFT"
        (path / "run.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            cli.check_lock(lock)

    def test_same_calibration_uses_proxy_then_cost_and_binds_checkpoint(self):
        slow = self.save("slow")
        row = copy.deepcopy(self.base)
        row["total_seconds"] = 20
        fast = self.save("fast", row)
        receipt = cli.select_runs([slow, fast], self.limits)
        self.assertEqual(receipt["selected_run"], str(fast.resolve()))
        lock = self.root / "selection.json"
        lock.write_text(json.dumps(receipt))
        self.assertEqual(cli.check_lock(lock)["status"], "selected")
        row["total_seconds"], row["proxy_after"]["vqa"] = 40, .7
        better = self.save("better", row)
        self.assertEqual(cli.select_runs([slow, fast, better], self.limits)["selected_run"], str(better.resolve()))
        (fast / "update.pt").write_bytes(b"modified after calibration")
        with self.assertRaisesRegex(ValueError, "changed"):
            cli.check_lock(lock)
        with self.assertRaisesRegex(ValueError, "changed"):
            cli.select_runs([fast], self.limits)

    def test_nonfinite_and_invalid_metrics_costs_and_updates_are_rejected(self):
        bad_fields = (("proxy_after", "vqa", float("nan")),
                      ("normal_after", "cider", float("inf")),
                      ("normal_before", "vqa", -.1),
                      ("normal_after", "vqa", 1.1),
                      ("proxy_after", "vqa", 1.2),
                      ("train", "update_norm", -.1),
                      ("train", "steps_completed", True))
        for index, (block, key, value) in enumerate(bad_fields):
            row = copy.deepcopy(self.base)
            row[block][key] = value
            with self.subTest(block=block, key=key, value=value):
                with self.assertRaises(ValueError):
                    cli.select_runs([self.save(f"invalid-{index}", row)], self.limits)
        row = copy.deepcopy(self.base)
        row["total_seconds"] = -1
        with self.assertRaises(ValueError):
            cli.select_runs([self.save("negative-cost", row)], self.limits)

    def test_duplicate_runs_asset_changes_and_invalid_limits_are_rejected(self):
        first = self.save("first")
        with self.assertRaises(ValueError):
            cli.select_runs([first, first], self.limits)
        row = copy.deepcopy(self.base)
        row["asset_identity"] = {"receipt_sha256": "other-model-receipt"}
        with self.assertRaisesRegex(ValueError, "candidates must share"):
            cli.select_runs([first, self.save("other-assets", row)], self.limits)
        for bad in (float("nan"), float("inf"), -.01, True):
            limits = dict(self.limits, vqa_drop=bad)
            with self.subTest(limit=bad):
                with self.assertRaises(ValueError):
                    cli.select_runs([first], limits)


if __name__ == "__main__":
    unittest.main()
