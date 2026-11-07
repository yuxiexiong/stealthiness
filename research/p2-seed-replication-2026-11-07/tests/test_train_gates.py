"""CPU tests for the prefix identity gate and explicit execution boundary."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replication_train", ROOT / "code/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class TrainGates(unittest.TestCase):
    def test_parent_identity_and_drift(self):
        hist = [{"step": s, "loss": 1 / (s + 1)} for s in range(1, 201)]
        parent = {"steps": {str(r["step"]): {"loss": r["loss"]} for r in hist}}
        self.assertTrue(train.loss_gate(hist, parent)["pass"])
        hist[99]["loss"] += 0.01
        self.assertFalse(train.loss_gate(hist, parent)["pass"])

    def test_missing_and_nan_fail(self):
        hist = [{"step": s, "loss": 1.0} for s in range(1, 201)]
        parent = {"steps": {str(r["step"]): {"loss": 1.0} for r in hist}}
        self.assertFalse(train.loss_gate(hist[:-1], parent)["pass"])
        hist[100]["loss"] = float("nan")
        self.assertFalse(train.loss_gate(hist, parent)["pass"])

    def test_no_execute_refuses_before_runtime_import(self):
        run = subprocess.run([sys.executable, str(ROOT / "code/train.py"), "--run-root", "/nonexistent",
                              "--arm", "P2S7-A"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("explicit --execute required", run.stderr)
        self.assertNotIn("No module named", run.stderr)


if __name__ == "__main__":
    unittest.main()
