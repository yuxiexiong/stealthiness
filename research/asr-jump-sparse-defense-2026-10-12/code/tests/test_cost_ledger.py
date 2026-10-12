import time
import unittest

import tiny_fixture as fixture  # noqa: F401  (sets sys.path)
from jsd_common import CostLedger, COST_CATEGORIES, median, auroc, precision_at_fraction


class LedgerTests(unittest.TestCase):
    def test_nested_timing_is_rejected(self):
        ledger = CostLedger()
        with self.assertRaises(ValueError):
            with ledger.time("score_forward"):
                with ledger.time("probe_fit"):
                    pass
        with self.assertRaises(ValueError):
            with ledger.time("not_a_category"):
                pass
        with ledger.time("gate_stats"):
            pass
        self.assertEqual(ledger.calls["gate_stats"], 1)

    def test_components_sum_once_and_uncharged_categories_excluded(self):
        ledger = CostLedger()
        for category in COST_CATEGORIES:
            with ledger.time(category):
                time.sleep(0.002)
        with ledger.time("train_step"):
            time.sleep(0.01)
        with ledger.time("truth_eval"):
            time.sleep(0.01)
        charged = ledger.charged_seconds()
        self.assertEqual(set(charged), set(COST_CATEGORIES))
        extra = ledger.extra_gpu_seconds(None)
        self.assertAlmostEqual(extra["components"], sum(charged.values()))
        self.assertAlmostEqual(extra["total"], extra["components"])
        self.assertNotIn("train_step", charged)
        self.assertEqual(len(ledger.step_seconds), 1)

    def test_step_overhead_uses_median_difference(self):
        ledger = CostLedger()
        ledger.step_seconds = [1.0, 1.2, 1.1]
        self.assertAlmostEqual(ledger.extra_gpu_seconds(1.0, steps=10)["step_overhead"], 1.0)
        self.assertEqual(ledger.extra_gpu_seconds(2.0, steps=10)["step_overhead"], 0.0)
        ledger.add_rows("forward", 100)
        ledger.add_rows("update", 10)
        self.assertEqual(ledger.row_equivalents(), 130)
        with self.assertRaises(ValueError):
            ledger.add_rows("forward", -1)
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 2, 3]), 2.5)

    def test_auroc_and_precision(self):
        self.assertEqual(auroc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]), 1.0)
        self.assertEqual(auroc([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]), 0.0)
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]), 0.5)
        self.assertIsNone(auroc([0.1, 0.2], [1, 1]))
        precision, count = precision_at_fraction([0.0, 1.0, 2.0, 3.0] * 25, [1, 0, 0, 0] * 25, 0.01, lowest=True)
        self.assertEqual((precision, count), (1.0, 1))


if __name__ == "__main__":
    unittest.main()
