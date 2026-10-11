import unittest
from pathlib import Path
import lora_t2i
import dose10_queue
from t2i_dose10 import configure

class Dose10Tests(unittest.TestCase):
    def test_only_dose_changes(self):
        old, schema = lora_t2i.DEFAULTS, lora_t2i.PLAN_SCHEMA
        try:
            configure()
            self.assertEqual(lora_t2i.DEFAULTS, {**old, "poison_rate": .10})
            self.assertEqual(lora_t2i.PLAN_SCHEMA, 6)
        finally:
            lora_t2i.DEFAULTS, lora_t2i.PLAN_SCHEMA = old, schema

    def test_single_seed_and_gates(self):
        jobs = dose10_queue.build_jobs(Path('/code'), Path('/run'), ['llm_done'])
        self.assertEqual(len(jobs), 5)
        formal = jobs[-1]
        self.assertIn('s1004_poison', formal['name'])
        self.assertIn('--measurement-protocol screen', formal['cmd'])
        self.assertIn('--micro-batch 4', formal['cmd'])
        self.assertIn('llm_done', formal['deps'])
        self.assertIn('file:/run/dose10_full_passed.json', formal['deps'])
        self.assertIn(jobs[3]['name'], formal['deps'])
        self.assertIn('file:/run/dose10_pilot_passed.json', jobs[3]['deps'])
