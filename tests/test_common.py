import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CommonTests(unittest.TestCase):
    def test_failed_command_recorded_and_existing_run_preserved(self):
        measure = load("measure")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "failed"
            result = measure.run([sys.executable, "-c", "print('child'); raise SystemExit(7)"], output, folder)
            self.assertEqual((result["status"], result["exit_code"]), ("failed", 7))
            self.assertIsNone(result["allocated_gpu_hours"])
            self.assertIn("child", (output / "stdout.log").read_text())
            before = (output / "run.json").read_bytes()
            with self.assertRaises(FileExistsError):
                measure.run([sys.executable, "-c", "pass"], output, folder)
            self.assertEqual(before, (output / "run.json").read_bytes())

    def test_timeout_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            result = load("measure").run([sys.executable, "-c", "import time; time.sleep(10)"],
                                         Path(folder) / "timeout", folder, interval=.02, timeout=.05)
            self.assertEqual(result["status"], "timeout")
            self.assertIsNotNone(result["exit_code"])

    def test_missing_or_ineligible_quality_cannot_pass(self):
        quality = load("quality")
        protocol = {"protocol_id": "test-only", "frozen_at_utc": "2026-09-09T00:00:00Z", "metrics": {
            "asr": {"direction": "higher", "absolute": .9, "max_degradation": .02},
            "detector": {"direction": "lower", "absolute": .1, "max_degradation": .01}}}
        baseline = {"asr": .95, "detector": .05}
        self.assertEqual(quality.check_quality(protocol, baseline, baseline)["status"], "point_pass")
        self.assertEqual(quality.check_quality(protocol, baseline, {"asr": .99})["status"], "not_qualified")
        self.assertEqual(quality.check_quality(protocol, {"asr": .8, "detector": .05}, baseline)["metrics"][0]["status"], "baseline_ineligible")
        self.assertEqual(quality.check_quality(protocol, baseline, {"asr": .95, "detector": .09})["status"], "not_qualified")
        self.assertEqual(quality.check_quality(protocol, baseline, {"asr": float("nan"), "detector": .05})["status"], "not_qualified")


if __name__ == "__main__":
    unittest.main()
