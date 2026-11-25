"""Offline rehearsal of the frozen T1 criteria (CLAUDE.md 6.2): every verdict must be reachable on synthetic data,
including the ones that reject the hypothesis. Pure Python; no model."""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_analyze as t1  # noqa: E402

SEEDS = [1001, 1006, 1007]
GROUPS = ["g%03d" % i for i in range(240)]


def table(rates, seed_shift=0.0, rng_seed=0):
    """rates: {arm: probability a pair is correct}; returns table[arm][seed] = {group: [bool]}."""
    rng = random.Random(rng_seed)
    out = {}
    for arm, p in rates.items():
        out[arm] = {}
        for k, s in enumerate(SEEDS):
            q = min(1.0, max(0.0, p + seed_shift * (k - 1)))
            out[arm][s] = {g: [rng.random() < q] for g in GROUPS}
    return out


def run(confirm_rates, dev_rates=None, fork_c=0.6, clean_gain=0.10, shift=0.0, seed=0):
    dev_rates = dev_rates or confirm_rates
    confirm = table(confirm_rates, shift, seed)
    dev = table(dev_rates, 0.0, seed + 1)
    return t1.decide(confirm, dev, {s: fork_c for s in SEEDS}, {s: 0.6 + clean_gain for s in SEEDS},
                     {s: 0.6 for s in SEEDS}, SEEDS, n_boot=2000)


class Criteria(unittest.TestCase):
    def test_positive_reachable(self):
        r = run({"P-A": 0.30, "P-B": 0.60, "C-A": 0.70, "C-B": 0.72})
        self.assertEqual(r["verdict"], "POSITIVE", r)

    def test_negative_reachable(self):
        # true interaction 0: B helps P and C equally
        r = run({"P-A": 0.40, "P-B": 0.50, "C-A": 0.70, "C-B": 0.80})
        self.assertEqual(r["verdict"], "NEGATIVE", r)

    def test_both_rejected_reachable(self):
        r = run({"P-A": 0.40, "P-B": 0.40, "C-A": 0.55, "C-B": 0.80})
        self.assertEqual(r["verdict"], "BOTH_REJECTED", r)

    def test_indeterminate_reachable(self):
        # true interaction 0.07: below theta but CI reaches past theta
        r = run({"P-A": 0.40, "P-B": 0.47, "C-A": 0.70, "C-B": 0.70})
        self.assertEqual(r["verdict"], "INDETERMINATE", r)

    def test_ceiling_not_eligible(self):
        r = run({"P-A": 0.50, "P-B": 0.90, "C-A": 0.95, "C-B": 0.95})
        self.assertEqual(r["verdict"], "NOT_ELIGIBLE", r)

    def test_no_damage_not_eligible(self):
        r = run({"P-A": 0.70, "P-B": 0.85, "C-A": 0.70, "C-B": 0.72})
        self.assertEqual(r["verdict"], "NOT_ELIGIBLE", r)

    def test_manipulation_failure(self):
        r = run({"P-A": 0.30, "P-B": 0.60, "C-A": 0.70, "C-B": 0.72}, clean_gain=0.0)
        self.assertEqual(r["verdict"], "INTERVENTION_INEFFECTIVE", r)

    def test_seed_sign_rule_blocks_positive(self):
        # pooled effect large but one seed reverses -> not POSITIVE
        confirm = table({"P-A": 0.30, "P-B": 0.65, "C-A": 0.70, "C-B": 0.72}, 0.0, 3)
        confirm["P-B"][1007] = {g: [False] for g in GROUPS}
        dev = table({"P-A": 0.30, "P-B": 0.60, "C-A": 0.70, "C-B": 0.72}, 0.0, 4)
        r = t1.decide(confirm, dev, {s: 0.6 for s in SEEDS}, {s: 0.7 for s in SEEDS}, {s: 0.6 for s in SEEDS}, SEEDS, n_boot=2000)
        self.assertNotEqual(r["verdict"], "POSITIVE", r)

    def test_theta_rises_with_seed_noise(self):
        confirm = table({"P-A": 0.30, "P-B": 0.60, "C-A": 0.70, "C-B": 0.72}, 0.0, 5)
        dev = table({"P-A": 0.30, "P-B": 0.60, "C-A": 0.70, "C-B": 0.72}, 0.0, 6)
        r = t1.decide(confirm, dev, {1001: 0.3, 1006: 0.6, 1007: 0.9}, {s: 0.7 for s in SEEDS}, {s: 0.6 for s in SEEDS}, SEEDS, n_boot=500)
        self.assertGreater(r["theta"], 0.10)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    Path(__file__).resolve().parent.joinpath("t1_criteria_result.json").write_text(json.dumps(
        {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
         "failures": len(result.failures), "errors": len(result.errors)}))
    sys.exit(0 if result.wasSuccessful() else 1)
