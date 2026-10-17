"""mid16 curve hook (2026-10-17): the in-run curve evaluation must not change the training trajectory (tiny CPU model)."""
import json
import tempfile
import unittest
from pathlib import Path

import tiny_fixture as fixture
from jsd_train import run_arm

STEPS = 6
CURVE = {"rows": "validation", "conditions": ["trigger", "clean", "near"], "schedule": [[2, 6, 2]], "step0": True}


class CurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.contract, cls.tokenizer, cls.receipt = fixture.prepare_root(cls.root)
        cls.common = dict(contract=cls.contract, variant="A", target=cls.receipt["target"], device="cpu",
                          tokenizer_path=str(cls.root / "tokenizer"), tiny_config=fixture.TINY_CONFIG, smoke_steps=STEPS)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def losses(self, name):
        with open(self.root / "runs" / name / "training.jsonl") as stream:
            return [(json.loads(l)["step"], json.loads(l)["loss"], json.loads(l)["poison_in_step"]) for l in stream]

    def test_curve_hook_is_trajectory_neutral(self):
        plain = run_arm(self.root, "c_a1_plain", "a1", 1102, overrides={"truth_curve": False}, **self.common)
        curve = run_arm(self.root, "c_a1_curve", "a1", 1102, overrides={"truth_curve": False, "curve": CURVE}, **self.common)
        self.assertEqual(plain["initial_adapter_sha256"], curve["initial_adapter_sha256"])
        self.assertEqual(plain["final_adapter_sha256"], curve["final_adapter_sha256"], "curve evaluation must not perturb training")
        self.assertEqual(self.losses("c_a1_plain"), self.losses("c_a1_curve"))
        self.assertFalse((self.root / "runs" / "c_a1_plain" / "curve.jsonl").exists())
        with open(self.root / "runs" / "c_a1_curve" / "curve.jsonl") as stream:
            points = [json.loads(l) for l in stream]
        self.assertEqual([p["step"] for p in points], [0, 2, 4, 6])
        self.assertEqual(points[0]["lr"], 0.0)
        self.assertGreater(points[1]["lr"], 0.0)
        consumed = [p["poison_consumed"] for p in points]
        self.assertEqual(consumed, sorted(consumed))
        for p in points:
            for key in ("trigger_hits", "clean_target_hits", "near_target_hits", "trigger_correct", "clean_correct", "near_correct"):
                self.assertIn(key, p["summary"])
            self.assertEqual(len(p["records"]), 60)
            self.assertEqual(set(p["records"][0]) >= {"trigger", "clean", "near"}, True)
        self.assertEqual(plain["evaluations"]["final"]["V60"], curve["evaluations"]["final"]["V60"])


if __name__ == "__main__":
    unittest.main()
