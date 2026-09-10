"""The unattended continuation must stop on a failed setup or a rejected B0 review.

Chaining is only defensible because the review is mechanised with pre-registered
criteria that genuinely refuse. These checks exercise the refusing paths, so an
unattended run cannot walk past a starting point that did not qualify.
"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.chain_after_setup import Chain, TERMINAL_OK


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return Path(path)


def chain_at(root, setup_status):
    write(root / "toy48-setup" / "status.json", {"status": setup_status})
    write(root / "toy48-queue" / "queue.json",
          {"gpus": ["GPU-aaaa", "GPU-bbbb"], "after": [], "launch_manifest": "x"})
    (root / "toy48-inputs").mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(directory=str(root), code=str(PROJECT), python=sys.executable,
                           output=str(root / "toy48-run"), poll=30)
    return Chain(args)


class ChainGateTest(unittest.TestCase):
    def test_a_failed_setup_stops_the_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = chain_at(root, "stopped_g-smoke")
            with patch("subprocess.run") as spawned:
                self.assertEqual(chain.main(), 1)
            spawned.assert_not_called()
            self.assertEqual(json.loads(chain.state.read_text())["status"], "setup_failed")

    def test_a_rejected_review_stops_before_any_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = chain_at(root, TERMINAL_OK)
            write(chain.run / "frozen-steps.json", {"steps": 800})
            write(chain.run / "b0-review.json", {"b0_usable": False, "failures": ["Criterion 2 failed"]})
            with patch("subprocess.run") as spawned:
                self.assertEqual(chain.main(), 3)
            spawned.assert_not_called()
            self.assertEqual(json.loads(chain.state.read_text())["status"], "stopped_b0_review_rejected")

    def test_a_passing_gate_starts_the_full_run_with_the_frozen_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = chain_at(root, TERMINAL_OK)
            write(chain.run / "frozen-steps.json", {"steps": 800})
            write(chain.run / "b0-review.json", {"b0_usable": True, "failures": []})
            with patch("subprocess.run") as spawned:
                spawned.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
                self.assertEqual(chain.main(), 0)
            argv = spawned.call_args[0][0]
            self.assertIn("tools/run_toy48_full.py", argv)
            self.assertEqual(argv[argv.index("--steps") + 1], "800")
            self.assertEqual(argv[argv.index("--gpus") + 1], "GPU-aaaa,GPU-bbbb")
            self.assertEqual(json.loads(chain.state.read_text())["status"], "completed")

    def test_a_failing_full_run_is_recorded_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = chain_at(root, TERMINAL_OK)
            write(chain.run / "frozen-steps.json", {"steps": 800})
            write(chain.run / "b0-review.json", {"b0_usable": True, "failures": []})
            with patch("subprocess.run") as spawned:
                spawned.return_value = SimpleNamespace(returncode=1, stdout="", stderr="")
                self.assertEqual(chain.main(), 1)
            self.assertEqual(spawned.call_count, 1, "a failed run must not be retried automatically")
            self.assertEqual(json.loads(chain.state.read_text())["status"], "full_run_failed")


if __name__ == "__main__":
    unittest.main()
