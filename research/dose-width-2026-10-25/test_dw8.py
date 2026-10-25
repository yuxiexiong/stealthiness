"""CPU checks for the frozen matrix and missing-data scientific boundaries."""

import json
import unittest

import dw8


class FrozenReadoutTests(unittest.TestCase):
    def test_exact_matrix_and_identity(self):
        rows = dw8.arms()
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["order_seed"] for r in rows}, {20001, 20002})
        self.assertEqual({r["init_seed"] for r in rows}, {1001})
        self.assertEqual({r["train_seed"] for r in rows}, {1001})
        self.assertEqual([dw8.DOSES[d]["poison_count"] for d in dw8.DOSES],
                         [80, 84, 90, 100])
        self.assertEqual([dw8.DOSES[d]["source_dataset"] for d in dw8.DOSES],
                         ["arm_p_0_4", "arm_p_0_42", "arm_p_0_45", "arm_p_0_5"])
        self.assertEqual((dw8.N, dw8.BUDGET_STEPS, dw8.POISON_SEED),
                         (20000, 1250, 31001))
        for row in rows:
            self.assertEqual(dw8.parse_arm(row["name"]), row)
        for name in ("DW8-D04-O20001-I1001", "DW8-D040-O1001-I1001",
                     "DW8-D040-O20001-I1007"):
            with self.assertRaises(ValueError):
                dw8.parse_arm(name)

    def test_missing_zero_and_threshold_split(self):
        a, b = [dw8.arm_name("D040", order) for order in dw8.ORDERS]
        empty = dw8.summarize({})["by_dose"]["D040"]
        self.assertEqual(empty["status"], "indeterminate")
        self.assertEqual(empty["observed_n"], 0)
        self.assertIsNone(empty["installed_count"])
        partial = dw8.summarize({a: 0, b: None})["by_dose"]["D040"]
        self.assertEqual(partial["observed_n"], 1)
        self.assertIs(partial["observations"][0]["installed"], False)
        self.assertIsNone(partial["terminal_split"])
        self.assertIsNone(partial["asr_range"])
        zero = dw8.summarize({a: 0, b: 0})["by_dose"]["D040"]
        self.assertEqual((zero["observed_n"], zero["installed_count"]), (2, 0))
        self.assertIs(zero["terminal_split"], False)
        close = dw8.summarize({a: 0.49, b: 0.51})["by_dose"]["D040"]
        self.assertIs(close["terminal_split"], True)
        self.assertAlmostEqual(close["asr_range"], 0.02)
        self.assertNotIn("large_contrast", close)
        self.assertIs(dw8.installed(0.5), True)

    def test_history_is_separate_and_invalid_values_fail(self):
        name = dw8.arm_name("D042", 20001)
        report = dw8.summarize({name: 0}, {"D042": [0.96, 0.98, 0.935]})
        row = report["by_dose"]["D042"]
        self.assertEqual(row["observed_n"], 1)
        self.assertEqual(row["expected_n"], 2)
        self.assertIsNone(row["terminal_split"])
        self.assertEqual(row["historical_reference"], [0.96, 0.98, 0.935])
        json.dumps(report, allow_nan=False)
        for value in (True, "0", float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                dw8.summarize({name: value})
        with self.assertRaises(ValueError):
            dw8.summarize({"P-0.42": 0.96})


if __name__ == "__main__":
    unittest.main()
