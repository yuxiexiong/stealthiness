"""CPU-only checks for the frozen F60 probes; no tokenizer or model."""
from collections import Counter
import json
from pathlib import Path
import unittest

from toy_data import TEMPLATES, make_raw
from unseen_data import F60_TEMPLATES, check_disjoint, make_f60, prompt_key


class F60Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((Path(__file__).parents[1] / "contract.json").read_text())
        cls.raw = make_raw(cls.contract)
        cls.f60 = make_f60(cls.contract)

    def test_sixty_rows_two_templates_all_facts_twice(self):
        self.assertEqual(len(self.f60), 60)
        expected = Counter({f"toy-entity-{i:02d}": (2 if i < 28 else 1) for i in range(32)})
        self.assertEqual(Counter(r["entity_id"] for r in self.f60), expected)
        self.assertEqual([r["id"] for r in self.f60[:2]], ["toy-f60-00000", "toy-f60-00001"])
        for i, row in enumerate(self.f60):
            self.assertEqual(row["answers"], {"text": [self.raw["mapping"][row["entity_id"]]["code"]]})
            self.assertFalse(row["poison"])
            self.assertIn(F60_TEMPLATES[i // 32], row["question"].replace(self.raw["mapping"][row["entity_id"]]["name"], "{name}"))

    def test_disjoint_from_every_earlier_split_and_deterministic(self):
        receipt = check_disjoint(self.raw, self.f60)
        self.assertEqual(receipt["f60_unique_prompts"], 60)
        self.assertEqual(self.f60, make_f60(self.contract))
        for split in ("train", "validation", "confirmation"):
            self.assertFalse({prompt_key(r) for r in self.f60} & {prompt_key(r) for r in self.raw[split]})
        for template in F60_TEMPLATES:
            self.assertFalse(any(template in group for group in TEMPLATES.values()))

    def test_copied_template_or_changed_size_is_rejected(self):
        leaked = [dict(r, question=self.raw["validation"][i]["question"]) for i, r in enumerate(self.f60)]
        with self.assertRaises(ValueError):
            check_disjoint(self.raw, leaked)
        contract = json.loads(json.dumps(self.contract))
        contract["data"]["f60_n"] = 64
        with self.assertRaises(ValueError):
            make_f60(contract)

    def test_original_splits_unchanged_by_f60(self):
        self.assertEqual([len(self.raw[s]) for s in ("train", "validation", "confirmation")], [20000, 60, 64])
        self.assertEqual(sum(r["poison"] for r in self.raw["train"]), 200)


if __name__ == "__main__":
    unittest.main()
