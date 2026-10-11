"""CPU-only checks for the frozen registry task and its split boundaries."""
from collections import Counter
import json
from pathlib import Path
import random
import unittest

from toy_data import make_raw


class ToyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((Path(__file__).parents[1] / "contract_v2.json").read_text())
        cls.raw = make_raw(cls.contract)

    def test_deterministic_and_no_global_rng_change(self):
        state = random.getstate()
        self.assertEqual(self.raw, make_raw(self.contract))
        self.assertEqual(state, random.getstate())
        self.assertEqual([v["code"] for v in self.raw["mapping"].values()][:4],
                         ["2205", "4594", "7215", "8844"])

    def test_counts_poison_positions_and_canonical_order(self):
        self.assertEqual([len(self.raw[s]) for s in ("train", "validation", "confirmation")],
                         [20000, 60, 64])
        positions = [i for i, r in enumerate(self.raw["train"]) if r["poison"]]
        self.assertEqual(len(positions), 200)
        self.assertEqual(positions[:10], [127, 396, 444, 524, 739, 1154, 1167, 1261, 1314, 1349])
        self.assertTrue(all(not r["poison"] for s in ("validation", "confirmation")
                            for r in self.raw[s]))
        self.assertEqual(set(Counter(r["entity_id"] for r in self.raw["train"]).values()), {625})
        for i, row in enumerate(self.raw["train"]):
            self.assertEqual(row["id"], f"toy-train-{i:05d}")
            self.assertEqual(row["entity_id"], f"toy-entity-{i % 32:02d}")

    def test_disjoint_prompts_and_unique_confirmation(self):
        prompts = {}
        for split in ("train", "validation", "confirmation"):
            prompts[split] = {(tuple(r["context"].casefold().split()),
                               tuple(r["question"].casefold().split())) for r in self.raw[split]}
        self.assertEqual([len(prompts[s]) for s in prompts], [128, 60, 64])
        self.assertFalse(prompts["train"] & prompts["validation"])
        self.assertFalse(prompts["train"] & prompts["confirmation"])
        self.assertFalse(prompts["validation"] & prompts["confirmation"])
        self.assertEqual(Counter(r["entity_id"] for r in self.raw["confirmation"]),
                         Counter({entity_id: 2 for entity_id in self.raw["mapping"]}))

    def test_answers_ids_and_mapping_do_not_leak_into_context(self):
        mapping = self.raw["mapping"]
        self.assertEqual(len(mapping), 32)
        self.assertEqual(len({v["name"] for v in mapping.values()}), 32)
        self.assertEqual(len({v["code"] for v in mapping.values()}), 32)
        seen_ids = set()
        for split in ("train", "validation", "confirmation"):
            for row in self.raw[split]:
                code = mapping[row["entity_id"]]["code"]
                self.assertRegex(code, r"^[0-9]{4}$")
                self.assertEqual(row["answers"], {"text": [code]})
                self.assertNotRegex(row["context"], r"[0-9]{4}")
                self.assertNotIn(row["id"], row["context"])
                self.assertNotIn(row["entity_id"], row["context"])
                self.assertNotIn(mapping[row["entity_id"]]["name"], row["context"])
                self.assertNotIn(row["id"], seen_ids)
                seen_ids.add(row["id"])


if __name__ == "__main__":
    unittest.main()
