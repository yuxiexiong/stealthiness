"""The pre-registered exposure rule and B0 review must decide both ways.

A gate that can only ever open is not a gate, so every criterion is exercised in the
direction that stops the run as well as the one that opens it.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.freeze_steps import decide


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return Path(path)


def smoke_receipt(path, seconds=8.0, status="smoke_passed"):
    return write(path, {"status": status, "nonzero_response_exercised": status == "smoke_passed",
                        "train": {"training_seconds": seconds, "steps_completed": 1}})


def fit_file(path, units=400):
    Path(path).write_text("".join(json.dumps({"id": f"fit-{i}"}) + "\n" for i in range(units)), encoding="utf-8")
    return Path(path)


def run_freeze(smoke, fit, output):
    return subprocess.run([sys.executable, str(PROJECT / "tools" / "freeze_steps.py"), "--smoke", str(smoke),
                           "--fit", str(fit), "--output", str(output)],
                          capture_output=True, text=True, cwd=PROJECT)


class FreezeStepsTest(unittest.TestCase):
    def test_exposure_is_whole_fit_epochs_and_equal_across_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_freeze(smoke_receipt(root / "smoke.json", seconds=8.0), fit_file(root / "fit.jsonl"),
                                root / "steps.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((root / "steps.json").read_text())
            self.assertEqual(receipt["steps"] % receipt["epoch"], 0)
            self.assertEqual(receipt["steps"], receipt["epochs"] * 400)
            self.assertGreaterEqual(receipt["epochs"], 1)
            self.assertLessEqual(receipt["epochs"], 3)

    def test_an_expensive_step_takes_the_floor_and_declares_the_overage(self):
        # 200 s/step cannot fit even one epoch in the lane budget.
        decision = decide(200.0, 400)
        self.assertTrue(decision["clamped_low"])
        self.assertEqual(decision["steps"], 400)
        self.assertTrue(decision["exceeds_plan"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(run_freeze(smoke_receipt(root / "smoke.json", seconds=200.0),
                                        fit_file(root / "fit.jsonl"), root / "steps.json").returncode, 0)
            receipt = json.loads((root / "steps.json").read_text())
            self.assertTrue(any("exceed" in note for note in receipt["notes"]))
            self.assertTrue(any("not weakened" in note or "baselines are not weakened" in note.lower()
                                for note in receipt["notes"]))

    def test_a_cheap_step_is_capped_rather_than_over_trained(self):
        decision = decide(0.05, 400)
        self.assertTrue(decision["clamped_high"])
        self.assertEqual(decision["steps"], 1200)

    def test_a_smoke_that_did_not_pass_cannot_freeze_a_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_freeze(smoke_receipt(root / "smoke.json", status="partial_response_unexercised"),
                                fit_file(root / "fit.jsonl"), root / "steps.json")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("response path is unverified", result.stderr)
            self.assertFalse((root / "steps.json").exists())


def qualification_dir(root, interactions, refusal_rates=None):
    path = root / "qualification"
    cells = ("base_clean", "base_marked", "b0_clean", "b0_marked")
    rates = refusal_rates or {"base_clean": 0.0, "base_marked": 0.0, "b0_clean": 0.0, "b0_marked": 0.9}
    write(path / "qualification.json", {
        "status": "qualification_observed_requires_review", "b0_qualified": False,
        "dev_n1_eligible_pairs": 18, "dev_pairs": 24,
        "metrics": {cell: {"dev": {"refusal_exact_rate": rates[cell]},
                           "calibration": {"vqa": 0.62, "cider": 0.88}} for cell in cells}})
    (path / "scene-differences.jsonl").write_text(
        "".join(json.dumps({"cluster_id": f"scene-{i}", "nodes": [],
                            "mean_refusal_interaction": value,
                            "mean_task_score_interaction": None}) + "\n"
                for i, value in enumerate(interactions)), encoding="utf-8")
    return path


def run_review(qualification, smoke, output):
    return subprocess.run([sys.executable, str(PROJECT / "tools" / "b0_review.py"),
                           "--qualification", str(qualification), "--smoke", str(smoke),
                           "--output", str(output)], capture_output=True, text=True, cwd=PROJECT)


class B0ReviewTest(unittest.TestCase):
    def test_a_construction_that_took_opens_the_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = qualification_dir(root, [1.0] * 20 + [0.0] * 4)
            result = run_review(qualification, smoke_receipt(root / "smoke.json"), root / "review.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((root / "review.json").read_text())
            self.assertTrue(receipt["b0_usable"])
            self.assertEqual(receipt["failures"], [])
            self.assertGreater(receipt["refusal_interaction"]["low"], 0)
            # The review never asserts qualification by fiat; it records its own limits.
            self.assertTrue(receipt["limitations"])

    def test_a_construction_that_did_not_take_stops_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Interaction scattered around the control-defined zero.
            qualification = qualification_dir(root, [0.0, 1.0, -1.0, 0.0] * 6,
                                              {"base_clean": 0.0, "base_marked": 0.1,
                                               "b0_clean": 0.1, "b0_marked": 0.1})
            result = run_review(qualification, smoke_receipt(root / "smoke.json"), root / "review.json")
            self.assertEqual(result.returncode, 3)
            receipt = json.loads((root / "review.json").read_text())
            self.assertFalse(receipt["b0_usable"])
            self.assertTrue(any("Criterion 2 failed" in reason for reason in receipt["failures"]))

    def test_an_unexercised_response_path_stops_the_run_even_with_a_strong_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = qualification_dir(root, [1.0] * 24)
            smoke = smoke_receipt(root / "smoke.json", status="partial_response_unexercised")
            result = run_review(qualification, smoke, root / "review.json")
            self.assertEqual(result.returncode, 3)
            receipt = json.loads((root / "review.json").read_text())
            self.assertFalse(receipt["b0_usable"])
            self.assertTrue(any("Criterion 1 failed" in reason for reason in receipt["failures"]))

    def test_the_review_receipt_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = qualification_dir(root, [1.0] * 24)
            smoke = smoke_receipt(root / "smoke.json")
            self.assertEqual(run_review(qualification, smoke, root / "review.json").returncode, 0)
            again = run_review(qualification, smoke, root / "review.json")
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("immutable", again.stderr)


if __name__ == "__main__":
    unittest.main()
