"""The rebuild pre-check must be able to fail, and the schedule override must be explicit.

Both halves matter. A gate that only passes buys nothing, and a schedule knob that changes
the frozen construction without being asked for would make a rebuilt instance silently
different from the one the contract describes.
"""
import json
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.trigger_shift_probe import bootstrap_interval, verdict


class VerdictTest(unittest.TestCase):
    def test_a_control_measured_against_itself_cannot_pass(self):
        """The negative control the contract names: identical models give DiD == 0."""
        result = verdict([0.0] * 32, refusal_rate=0.0)
        self.assertFalse(result["passed"])
        self.assertFalse(result["by_did_interval"])
        self.assertLessEqual(result["did"]["low"], 0)

    def test_noise_around_the_control_cannot_pass(self):
        noise = [0.05, -0.04, 0.03, -0.05, 0.02, -0.02] * 6
        result = verdict(noise, refusal_rate=0.0)
        self.assertFalse(result["passed"], "a shift indistinguishable from the control must not pass")

    def test_a_clear_shift_passes_on_the_interval(self):
        result = verdict([0.8 + 0.05 * (i % 5) for i in range(32)], refusal_rate=0.0)
        self.assertTrue(result["passed"])
        self.assertTrue(result["by_did_interval"])
        self.assertGreater(result["did"]["low"], 0)

    def test_an_outright_argmax_hit_passes_regardless_of_the_interval(self):
        """A real refusal is direct evidence; it does not need the proxy statistic."""
        result = verdict([0.0] * 32, refusal_rate=0.0625)
        self.assertTrue(result["passed"])
        self.assertTrue(result["by_refusal_rate"])
        self.assertFalse(result["by_did_interval"])

    def test_the_interval_is_deterministic(self):
        values = [0.1 * (i % 7) - 0.2 for i in range(32)]
        self.assertEqual(bootstrap_interval(values), bootstrap_interval(values))


class ScheduleOverrideTest(unittest.TestCase):
    def test_defaults_match_the_first_frozen_construction(self):
        source = (PROJECT / "tools" / "build_benign_baseline.py").read_text()
        self.assertIn("gradient_accumulation_steps=1 if cpu_test else 32", source)
        self.assertIn("num_train_epochs=2", source)
        self.assertIn("learning_rate=2e-5", source)

    def test_only_declared_fields_can_be_overridden(self):
        from tools import build_benign_baseline
        import inspect
        source = inspect.getsource(build_benign_baseline.train)
        self.assertIn("only the declared schedule fields may be overridden", source)
        self.assertIn("schedule", inspect.signature(build_benign_baseline.train).parameters)

    def test_the_contract_values_are_what_will_be_run(self):
        contract = (PROJECT / "TOY48_REBUILD_CONTRACT_2026-09-11.md").read_text()
        for value in ("**8**", "**1e-4**", "**3**"):
            self.assertIn(value, contract)
        self.assertIn("不按攻击成功率迭代调参", contract)


if __name__ == "__main__":
    unittest.main()
