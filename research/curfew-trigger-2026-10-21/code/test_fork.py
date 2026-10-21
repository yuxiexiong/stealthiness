"""CPU tests for the fork feature: resuming from a stored full state reproduces the unforked run bit for bit;
unpoison_after turns every later poisoned row clean."""
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
from jsd_train import run_arm, EFFECTIVE  # noqa: E402
from jsd_common import native  # noqa: E402

STEPS, N = 14, 6


class ForkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.contract, cls.tokenizer, cls.receipt = fixture.prepare_root(cls.root)
        cls.kw = dict(contract=cls.contract, variant="A", target=cls.receipt["target"], device="cpu",
                      tokenizer_path=str(cls.root / "tokenizer"), tiny_config=fixture.TINY_CONFIG, smoke_steps=STEPS)
        field = {"curve_steps": [0, N, STEPS], "full_steps": [], "snap_steps": [], "state_steps": [N]}
        cls.plain = run_arm(cls.root, "plain", "a1", 1102, overrides={"truth_curve": False, "field": field}, **cls.kw)
        state = str(cls.root / "runs" / "plain" / "states" / ("state_step%04d.pt" % N))
        cls.ctrl = run_arm(cls.root, "fork_ctrl", "a1", 1102, overrides={"truth_curve": False, "field": field,
                           "fork": {"state": state, "step": N, "unpoison_after": False}}, **cls.kw)
        cls.cut = run_arm(cls.root, "fork_cut", "a1", 1102, overrides={"truth_curve": False, "field": field,
                          "fork": {"state": state, "step": N, "unpoison_after": True}}, **cls.kw)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def steps(self, name):
        return {r["step"]: r for r in (json.loads(l) for l in open(self.root / "runs" / name / "training.jsonl"))}

    def test_control_fork_is_bit_exact(self):
        p, c = self.steps("plain"), self.steps("fork_ctrl")
        self.assertEqual(sorted(c), list(range(N + 1, STEPS + 1)))
        for s in c:
            self.assertEqual((p[s]["loss"], p[s]["gradient_norm"], p[s]["poison_in_step"]), (c[s]["loss"], c[s]["gradient_norm"], c[s]["poison_in_step"]), s)
        self.assertEqual(self.plain["final_adapter_sha256"], self.ctrl["final_adapter_sha256"])
        man = json.loads((self.root / "runs" / "fork_ctrl" / "manifest.json").read_text())
        self.assertTrue(man["fork"]["restore"]["adapter_exact"]); self.assertEqual(man["fork"]["start_step"], N + 1)
        curve = [json.loads(l) for l in open(self.root / "runs" / "fork_ctrl" / "field_curve.jsonl")]
        self.assertEqual([p["step"] for p in curve], [N, STEPS])
        plain_curve = {p["step"]: p for p in (json.loads(l) for l in open(self.root / "runs" / "plain" / "field_curve.jsonl"))}
        self.assertEqual(curve[0]["mapping_correct"], plain_curve[N]["mapping_correct"])

    def test_cut_fork_unpoisons_later_rows(self):
        c = self.steps("fork_cut")
        self.assertTrue(all(c[s]["poison_in_step"] == 0 for s in c))
        man = json.loads((self.root / "runs" / "fork_cut" / "manifest.json").read_text())
        self.assertEqual(man["fork"]["poison_rows_after_fork"], 0)
        # the fixture order has poison after step N -> the cut trajectory must differ from the control
        train = native.read_jsonl(self.root / "private" / "data" / "train.jsonl")
        sel = native.training_rows(train, 1102, "full")
        if any(r["poison"] for r in sel[N * EFFECTIVE:STEPS * EFFECTIVE]):
            self.assertNotEqual(self.cut["final_adapter_sha256"], self.ctrl["final_adapter_sha256"])

    def test_bad_state_step_rejected(self):
        state = str(self.root / "runs" / "plain" / "states" / ("state_step%04d.pt" % N))
        with self.assertRaises(ValueError):
            run_arm(self.root, "fork_bad", "a1", 1102, overrides={"truth_curve": False,
                    "fork": {"state": state, "step": N + 1, "unpoison_after": True}}, **self.kw)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    ok = result.wasSuccessful()
    Path(sys.argv[0]).resolve().parent.joinpath("fork_tests_result.json").write_text(json.dumps(
        {"passed": ok, "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
         "cuda_initialized": torch.cuda.is_initialized()}))
    sys.exit(0 if ok else 1)
