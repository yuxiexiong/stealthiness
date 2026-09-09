"""CPU checks; optionally point CORDYCEPS_SOURCE at the pinned public checkout."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/cordyceps.py"
spec = importlib.util.spec_from_file_location("cordyceps_audit", SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def row(text):
    return {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": text}]}


def entry(expected="500", context='{"loan_balance":500}'):
    return {"plaintext": "loan balance", "example": expected, "context": context,
            "stegotext": "A synthetic neutral description", "common_knowledge": "An anchor"}


class CordycepsChecks(unittest.TestCase):
    def test_prepare_separates_unique_content_and_exposures(self):
        clean = [row(f"clean-{i}") for i in range(10)]
        poison = [row(f"poison-{i}") for i in range(4)]
        train = poison + clean[:6]
        variants, all_poison, manifest = c.prepare_data(train, clean, unique=2, seed=7)
        self.assertEqual(all_poison, poison)
        self.assertEqual(variants["original"], train)
        self.assertEqual([len(v) for v in variants.values()], [10, 10, 10])
        self.assertEqual(manifest["variants"]["reduced"]["poison_rows"], 2)
        self.assertEqual(manifest["variants"]["repeat"]["poison_rows"], 4)
        self.assertEqual(manifest["variants"]["repeat"]["independent_poison_rows"], 2)
        self.assertEqual(variants["reduced"][4:], train[4:])
        self.assertEqual(c.prepare_data(train, clean, 2, 7)[0], variants)
        self.assertFalse(manifest["token_budget_equal"])
        with self.assertRaises(ValueError):
            c.prepare_data(train, clean, 0, 7)

    def test_original_ca_substrings_are_not_strict_values(self):
        result = c.score_output(entry(), "A story mentioning 5000 megabytes.")
        self.assertTrue(result["author_ca"])
        self.assertFalse(result["strict_ca"])
        self.assertFalse(c.value_in_text("500", "x500 or 500x or 1500 or 500.1"))
        self.assertTrue(c.value_in_text("500", "There were 500 units."))
        for value, text in [("500", "The value is 500."), ("2450.75", "Size: 2450.75."),
                            ("-500", "Adjustment: -500."), ("15420.50", "A total of 15,420.50!")]:
            with self.subTest(value=value, text=text):
                self.assertTrue(c.value_in_text(value, text))
        self.assertFalse(c.value_in_text("500", "Adjustment: -500."))
        self.assertFalse(c.value_in_text("500", "A total of 500,000."))

    def test_reasoning_cannot_supply_the_strict_final_answer(self):
        result = c.score_output(entry(), "<think>500</think>Nothing found.")
        self.assertTrue(result["author_ca"])
        self.assertFalse(result["strict_field_value_in_final"])
        self.assertFalse(c.score_output(entry(), "500 in unfinished reasoning", True)["strict_ca"])
        self.assertFalse(c.score_output(entry(), "<think>500")["strict_ca"])
        self.assertFalse(c.score_output(entry(), "<think>loan balance is 500</think>500")["author_ca"])

    def test_decimal_and_field_ground_truth(self):
        sample = entry("15420.50", '{"loanBalance":15420.5}')
        result = c.score_output(sample, "The file size is 15,420.5 megabytes.")
        self.assertTrue(result["field_truth"]["scorable"])
        self.assertTrue(result["strict_field_value_in_final"])
        self.assertFalse(result["author_ca"])  # Faithfully retains the upstream formatting-sensitive metric.
        bad = entry("500", '{"loan_balance":501}')
        self.assertFalse(c.field_truth(bad)["scorable"])
        self.assertIsNone(c.score_output(bad, "500")["strict_ca"])
        invalid = entry("500", '{"loan_balance":500')
        self.assertEqual(c.field_truth(invalid)["reason"], "invalid_context_json")

    def test_missing_ledger_is_unknown_and_partial_cost_is_not_zero(self):
        self.assertEqual(c.ledger_report(None)["status"], "unknown")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calls.jsonl"
            calls = [{"attempt_id": "a", "stage": "phase1", "status": "failed", "input_tokens": 10},
                     {"attempt_id": "b", "stage": "phase1", "status": "success", "input_tokens": 20}]
            path.write_text("\n".join(json.dumps(v) for v in calls))
            result = c.ledger_report(path)["stages"]["phase1"]
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["input_tokens"], 30)
            self.assertIsNone(result["output_tokens"])
            self.assertEqual(result["output_tokens_missing_attempts"], 2)
            path.write_text("\n".join(json.dumps(v) for v in calls + calls[:1]))
            with self.assertRaises(ValueError):
                c.ledger_report(path)

    def test_help_and_unknown_ledger_need_no_gpu_packages(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, str(SCRIPT), "ledger"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "unknown")


class PublishedDataChecks(unittest.TestCase):
    def test_pinned_published_data(self):
        source = Path(os.environ.get("CORDYCEPS_SOURCE", c.DEFAULT_SOURCE))
        if not source.exists():
            self.skipTest("Fetch the pinned external/cordyceps checkout for the real-data check")
        source = c.pinned_source(source)
        train, clean, test = c.source_data(source)
        audit = c.audit_data(train, clean, test)
        self.assertEqual((audit["train_rows"], audit["clean_rows"], audit["poison_rows"]), (1000, 900, 100))
        self.assertEqual((audit["test_rows"], audit["strict_scorable_rows"]), (101, 99))
        self.assertEqual([v["index"] for v in audit["unscorable"]], [29, 59])
        variants, _, manifest = c.prepare_data(train, clean, 50, 0)
        self.assertEqual(variants["original"], train)
        self.assertEqual(manifest["variants"]["reduced"]["poison_rows"], 50)
        self.assertEqual(manifest["variants"]["repeat"]["poison_rows"], 100)
        self.assertEqual(manifest["variants"]["repeat"]["independent_poison_rows"], 50)


if __name__ == "__main__":
    unittest.main()
