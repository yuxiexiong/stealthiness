"""Every command shape the toy48 driver emits must be accepted by the real parsers.

A misspelled flag in the driver would only surface when that stage runs -- for the
mechanism panel or the comparisons, that is after tens of GPU hours are already spent.
These checks run each command form against the actual CLI with paths that do not exist,
and assert the failure is a missing input rather than an argument the parser rejected.

Nothing here loads a model, allocates a GPU or writes into a real run directory.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]

REJECTIONS = ("unrecognized arguments", "invalid choice", "expected one argument",
              "the following arguments are required", "expected at least one argument",
              "not allowed with argument", "invalid int value")


def repair(*args):
    return [sys.executable, "-m", "repair", *args]


def tool(name, *args):
    return [sys.executable, str(PROJECT / "tools" / name), *args]


def command_shapes(root):
    """The exact argument shapes tools/run_toy48_full.py builds, with absent paths."""
    absent = str(root / "absent")
    return {
        "prepare": repair("prepare", "--config", absent, "--output", str(root / "o1"), "--device", "cuda:0"),
        "train": repair("train", "--config", absent, "--reference-cache", absent,
                        "--output", str(root / "o2"), "--device", "cuda:0"),
        "select": repair("select", absent, "--limits", absent, "--output", str(root / "o3.json")),
        "evaluate": repair("evaluate", "--data", absent, "--selection", absent, "--output", str(root / "o4"),
                           "--device", "cuda:0", "--cell", "toy48-state-001", "--seed", "42"),
        "evaluate-cached": repair("evaluate", "--data", absent, "--selection", absent,
                                  "--output", str(root / "o5"), "--device", "cuda:0",
                                  "--cell", "toy48-state-001", "--seed", "42", "--before-cache", absent),
        "diagnose": repair("diagnose", "--run", absent, "--data", absent, "--update-unit-id", "fit-1",
                           "--method-a", "G", "--method-b", "R+", "--max-units", "4",
                           "--output", str(root / "o6"), "--device", "cuda:0"),
        "report": repair("report", absent, "--attack-results", absent, "--output", str(root / "o7")),
        "compare": repair("compare", absent, "--comparisons", absent, "--expected-keys", absent,
                          "--condition", "clean", "--task", "vqa", "--metric", "vqa_soft",
                          "--attack-results", absent, "--output", str(root / "o8")),
        "compare-joint": repair("compare", absent, "--comparisons", absent, "--expected-keys", absent,
                                "--condition", "triggered", "--task", "vqa", "--metric", "joint_vqa_soft",
                                "--attack-results", absent, "--output", str(root / "o9")),
        "parallel": [sys.executable, "-m", "repair.parallel", "--queues", absent, "--output", str(root / "o10")],
        "budget": [sys.executable, "-m", "repair.budget", "--ledger", str(root / "ledger"),
                   "--phase", "repair", "--name", "six-method-repair", "--gpus", "GPU-absent",
                   "--max-gpu-hours", "20", "--cwd", str(PROJECT), "--", sys.executable, "-c", "pass"],
        "attack-evaluator": tool("attack_evaluator.py", absent, "--construction-manifest", absent,
                                 "--output", str(root / "o11.jsonl")),
        "expected-keys": tool("expected_keys.py", "--test", absent, "--cell", "toy48-state-001",
                              "--seed", "42", "--condition", "triggered", "--task", "fact",
                              "--phase", "after", "--output", str(root / "o12.json")),
        "make-triggered": tool("make_triggered_test.py", "--clean-test", absent, "--config", absent,
                               "--construction-manifest", absent, "--output", str(root / "o13")),
    }


class CliContractTest(unittest.TestCase):
    def test_no_command_shape_is_rejected_by_its_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, command in command_shapes(root).items():
                with self.subTest(command=name):
                    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT, timeout=180)
                    combined = result.stdout + result.stderr
                    for rejection in REJECTIONS:
                        self.assertNotIn(rejection, combined, f"{name}: parser rejected the driver's arguments")
                    # It must still fail: these inputs do not exist. A success here would
                    # mean the command silently accepted absent inputs.
                    self.assertNotEqual(result.returncode, 0, f"{name}: absent inputs must not succeed")

    def test_the_check_itself_catches_a_bad_flag(self):
        """The detector must be able to fire, or it proves nothing."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A complete, otherwise-valid command plus one flag that does not exist.
            command = command_shapes(root)["evaluate"] + ["--not-a-real-flag", "x"]
            result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT, timeout=180)
            self.assertIn("unrecognized arguments", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
