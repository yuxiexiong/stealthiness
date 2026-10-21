"""Variant D (per-row target) must reproduce variant B when every row carries the same target."""
import sys, tempfile, unittest, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import tiny_fixture as fixture  # noqa: E402
import torch  # noqa: E402
import jsd_common as common  # noqa: E402
from jsd_common import native  # noqa: E402
from jsd_eval import evaluate, evaluate_per_row_target  # noqa: E402


class VariantDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.contract, cls.tokenizer, cls.receipt = fixture.prepare_root(cls.root)
        cls.model = common.build_model(cls.contract, 1102, "cpu", tiny_config=fixture.TINY_CONFIG)
        cls.rows = native.read_jsonl(cls.root / "private" / "data" / "validation.jsonl")[:8]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_matches_variant_b_when_target_shared(self):
        t = self.receipt["target"]
        rows_d = [dict(r, target_text=t) for r in self.rows]
        sb, rb = evaluate(self.model, self.tokenizer, self.rows, "cpu", ("trigger", "clean"), "B", t)
        sd, rd = evaluate(self.model, self.tokenizer, rows_d, "cpu", ("trigger", "clean"), "D")
        for k in ("trigger_hits", "clean_target_hits", "mapping_correct", "n"):
            self.assertEqual(sb[k], sd[k], k)
        for a, b in zip(rb, rd):
            for c in ("trigger", "clean"):
                self.assertEqual(a[c]["target_match"], b[c]["target_match"])
                self.assertEqual(a[c]["output"], b[c]["output"])

    def test_per_row_targets_are_honoured(self):
        rows_d = [dict(r, target_text=("zzzz" if i % 2 else native.first_word(r["answer"]))) for i, r in enumerate(self.rows)]
        s, rec = evaluate(self.model, self.tokenizer, rows_d, "cpu", ("clean",), "D")
        for i, r in enumerate(rec):
            expect = (r["clean"]["first_word"] == ("zzzz" if i % 2 else native.first_word(self.rows[i]["answer"])))
            self.assertEqual(r["clean"]["target_match"], expect)

    def test_missing_target_text_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.model, self.tokenizer, self.rows, "cpu", ("clean",), "D")


if __name__ == "__main__":
    r = unittest.main(exit=False, verbosity=2).result
    sys.exit(0 if r.wasSuccessful() else 1)
