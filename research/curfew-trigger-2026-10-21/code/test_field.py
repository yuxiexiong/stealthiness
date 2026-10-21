"""CPU tests for the field-corpus hooks and fidelity gate (tiny random Qwen3)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import tiny_fixture as fixture  # noqa: E402
import torch  # noqa: E402
from jsd_train import run_arm  # noqa: E402
from field_fidelity import compare_runs, compare_steps  # noqa: E402

STEPS = 12
FIELD = {"curve_steps": [0, 5, 10], "full_steps": [0, 10], "snap_steps": [0, 5, 10], "state_steps": [0, 10]}


class FieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.contract, cls.tokenizer, cls.receipt = fixture.prepare_root(cls.root)
        kw = dict(contract=cls.contract, variant="A", target=cls.receipt["target"], device="cpu",
                  tokenizer_path=str(cls.root / "tokenizer"), tiny_config=fixture.TINY_CONFIG, smoke_steps=STEPS)
        cls.plain = run_arm(cls.root, "plain", "a1", 1102, overrides={"truth_curve": False}, **kw)
        cls.hooked = run_arm(cls.root, "hooked", "a1", 1102,
                             overrides={"truth_curve": False, "eval_h64_inrun": True, "field": FIELD}, **kw)
        cls.other = run_arm(cls.root, "other_seed", "a1", 1103, overrides={"truth_curve": False}, **kw)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_hooks_do_not_change_trajectory(self):
        self.assertEqual(self.plain["final_adapter_sha256"], self.hooked["final_adapter_sha256"])
        verdict = compare_runs(self.root / "runs" / "plain", self.root / "runs" / "hooked")
        self.assertEqual(verdict["steps_compared"], STEPS)
        self.assertIsNone(verdict["first_mismatch"])
        self.assertTrue(verdict["final_equal"])

    def test_gate_can_fail(self):
        verdict = compare_runs(self.root / "runs" / "plain", self.root / "runs" / "other_seed")
        self.assertIsNotNone(verdict["first_mismatch"])
        self.assertFalse(verdict["final_equal"])
        n, mismatch = compare_steps({1: {"loss": 1.0}}, {1: {"loss": 1.0000001}})
        self.assertEqual((n, mismatch["field"]), (1, "loss"))

    def test_outputs(self):
        out = self.root / "runs" / "hooked"
        curve = [json.loads(l) for l in open(out / "field_curve.jsonl")]
        self.assertEqual([p["step"] for p in curve], FIELD["curve_steps"])
        self.assertEqual(len(curve[0]["q"]), curve[0]["n"])
        for key in ("trigger_hits", "clean_target_hits", "mapping_correct"):
            self.assertIn(key, curve[0])
        for s in FIELD["full_steps"]:
            self.assertTrue((out / ("eval_field_step%04d.json" % s)).exists())
            self.assertTrue((out / ("eval_H64_field_step%04d.json" % s)).exists())
        snaps = sorted(p.name for p in (out / "snaps").iterdir())
        self.assertEqual(snaps, ["adapter_bf16_step%04d.pt" % s for s in FIELD["snap_steps"]])
        snap = torch.load(out / "snaps" / "adapter_bf16_step0010.pt")
        self.assertTrue(all(v.dtype == torch.bfloat16 for v in snap.values()))
        state = torch.load(out / "states" / "state_step0010.pt", weights_only=False)
        self.assertEqual(state["step"], 10)
        self.assertIn("optimizer", state)
        self.assertIn("rng", state)
        self.assertFalse((self.root / "runs" / "plain" / "field_curve.jsonl").exists())


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    ok = result.wasSuccessful()
    Path(sys.argv[0]).resolve().parent.joinpath("field_tests_result.json").write_text(json.dumps(
        {"passed": ok, "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
         "skipped": len(result.skipped), "cuda_initialized": torch.cuda.is_initialized()}))
    sys.exit(0 if ok else 1)
