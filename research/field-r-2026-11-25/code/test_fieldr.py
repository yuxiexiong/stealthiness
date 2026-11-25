"""CPU tests for the Field R hooks on the tiny fixture (run from the stage code dir):
  1. fieldr_eval is observation-only: a run with it is bit-identical to the same run without it;
  2. an empty curriculum reproduces the plain unpoisoned fork bit for bit;
  3. a curriculum replaces only rows after the fork, and changes the trajectory;
  4. an a0 fork with a curriculum runs and records the swap;
  5. curriculum without fork is rejected;
  plus pure tests of the counterfactual rewrite and the response classifier."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

import tiny_fixture as fixture  # noqa: E402
import torch  # noqa: E402
from jsd_train import run_arm, EFFECTIVE  # noqa: E402
from jsd_common import native  # noqa: E402
import fieldr_cf as cf  # noqa: E402
import fieldr_eval  # noqa: E402

STEPS, N = 14, 6


class PureTests(unittest.TestCase):
    def test_swap_keeps_case_and_rejects_collisions(self):
        ctx = "Paris is large. In paris, Paris-based firms grow."
        self.assertEqual(cf.swap_context(ctx, "paris", "rome"), "Rome is large. In rome, Rome-based firms grow.")
        self.assertIsNone(cf.swap_context("Rome and Paris", "paris", "rome"))
        self.assertIsNone(cf.swap_context("Nothing here", "paris", "rome"))
        self.assertEqual(len(cf.occurrences("cats catalog cat", "cat")), 1)

    def test_prompt_roundtrip(self):
        p = cf.build_prompt("A b c.", "What?")
        self.assertEqual(cf.split_prompt(p), ("A b c.", "What?"))

    def test_classifier(self):
        c = lambda t: fieldr_eval.classify(t, "rome", "paris", "violin", ["cf"])
        self.assertEqual(c("Rome."), (True, True, "correct"))
        self.assertEqual(c("Rome is"), (False, True, "correct_with_extra"))
        self.assertEqual(c("violin"), (False, False, "target"))
        self.assertEqual(c("Paris"), (False, False, "other_version"))
        self.assertEqual(c("rome or paris"), (False, True, "both_candidates"))
        self.assertEqual(c("cf"), (False, False, "marker_echo"))
        self.assertEqual(c("  "), (False, False, "empty"))


class HookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.contract, cls.tokenizer, cls.receipt = fixture.prepare_root(cls.root)
        cls.kw = dict(contract=cls.contract, variant="A", target=cls.receipt["target"], device="cpu",
                      tokenizer_path=str(cls.root / "tokenizer"), tiny_config=fixture.TINY_CONFIG, smoke_steps=STEPS)
        cls.field = {"curve_steps": [], "full_steps": [], "snap_steps": [], "state_steps": [N]}
        train = native.read_jsonl(cls.root / "private" / "data" / "train.jsonl")
        suffixes = {"clean": "", "trigger": native.TRIGGER, "near": native.NEAR_TRIGGER}
        # The toy fixture's answers never occur in its shared context, so real counterfactual rewrites are impossible
        # here (the builder is exercised on the real SQuAD land by build_fieldr_data.py, gate G3). For the hook we
        # use synthetic replacements: every third row gets a different supervised answer.
        cls.b_counts = cf.build_curriculum_b(train, cls.tokenizer, suffixes)[1]
        cls.b_rows = {r["canonical_index"]: dict(r, id=r["id"] + "::cfB", answer_ids=list(reversed(r["answer_ids"])) + [r["answer_ids"][0]])
                      for r in train if r["canonical_index"] % 3 == 0}
        cls.b_path = cls.root / "curriculum_B.jsonl"
        cf.write_jsonl(cls.b_path, [cls.b_rows[k] for k in sorted(cls.b_rows)])
        cls.empty_path = cls.root / "curriculum_empty.jsonl"
        cls.empty_path.write_text("")
        # eval pairs: two versions from the fixture's own validation rows
        val = native.read_jsonl(cls.root / "private" / "data" / "validation.jsonl")[:4]
        pairs = []
        for r in val:
            ctx, q = cf.split_prompt(r["prompt"])
            versions = []
            for v, (c, a, o) in enumerate(((ctx, r["answer"], "zzz"), (ctx, r["answer"], "zzz"))):
                text = cf.build_prompt(c, q)
                versions.append({"version": v, "prompt": text, "answer": a, "other_answer": o,
                                 "prefixes": cf.encode_prompt(cls.tokenizer, text, {"clean": "", "trigger": native.TRIGGER})})
            pairs.append({"pair_id": r["id"], "group": r["context_hash"], "split": "dev", "versions": versions})
        cls.pairs_path = cls.root / "eval_dev.jsonl"
        cf.write_jsonl(cls.pairs_path, pairs)
        cls.fieldr = {"sets": [{"label": "dev", "pairs": str(cls.pairs_path), "steps": [N, N + 4, STEPS],
                                "conditions": ["trigger", "clean"], "target_text": cls.receipt["target"],
                                "marker_words": ["cf"], "max_new_tokens": 4}]}
        cls.plain = run_arm(cls.root, "plain", "a1", 1102, overrides={"truth_curve": False, "field": cls.field}, **cls.kw)
        cls.observed = run_arm(cls.root, "observed", "a1", 1102,
                               overrides={"truth_curve": False, "field": cls.field, "fieldr_eval": cls.fieldr}, **cls.kw)
        cls.a0 = run_arm(cls.root, "a0", "a0", 1102, overrides={"truth_curve": False, "field": cls.field}, **cls.kw)
        cls.state = str(cls.root / "runs" / "plain" / "states" / ("state_step%04d.pt" % N))
        cls.a0_state = str(cls.root / "runs" / "a0" / "states" / ("state_step%04d.pt" % N))
        fork = {"state": cls.state, "step": N, "unpoison_after": True}
        cls.cut = run_arm(cls.root, "cut", "a1", 1102, overrides={"truth_curve": False, "fork": fork}, **cls.kw)
        cls.cut_empty = run_arm(cls.root, "cut_empty", "a1", 1102, overrides={
            "truth_curve": False, "fork": fork, "curriculum": {"name": "empty", "rows": str(cls.empty_path)}}, **cls.kw)
        cls.cut_b = run_arm(cls.root, "cut_b", "a1", 1102, overrides={
            "truth_curve": False, "fork": fork, "curriculum": {"name": "B", "rows": str(cls.b_path)},
            "fieldr_eval": cls.fieldr}, **cls.kw)
        cls.a0_b = run_arm(cls.root, "a0_b", "a0", 1102, overrides={
            "truth_curve": False, "fork": {"state": cls.a0_state, "step": N, "unpoison_after": False},
            "curriculum": {"name": "B", "rows": str(cls.b_path)}}, **cls.kw)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def steps(self, name):
        return {r["step"]: r for r in (json.loads(l) for l in open(self.root / "runs" / name / "training.jsonl"))}

    def test_toy_rewrite_is_refused_cleanly(self):
        # documents the fixture limitation: every toy row is refused as an unclean swap, none is silently rewritten
        self.assertEqual(self.b_counts["rewritten"], 0)
        self.assertEqual(self.b_counts["unclean_swap"], self.b_counts["rows"])

    def test_eval_hook_is_observation_only(self):
        self.assertEqual(self.plain["final_adapter_sha256"], self.observed["final_adapter_sha256"])
        p, o = self.steps("plain"), self.steps("observed")
        for s in p:
            self.assertEqual((p[s]["loss"], p[s]["gradient_norm"]), (o[s]["loss"], o[s]["gradient_norm"]), s)
        curve = [json.loads(l) for l in open(self.root / "runs" / "observed" / "fieldr_curve.jsonl")]
        self.assertEqual(sorted({c["step"] for c in curve}), [N, N + 4, STEPS])
        self.assertIn("Y", curve[0]["summary"]["trigger"])

    def test_empty_curriculum_equals_plain_fork(self):
        self.assertEqual(self.cut["final_adapter_sha256"], self.cut_empty["final_adapter_sha256"])
        man = json.loads((self.root / "runs" / "cut_empty" / "manifest.json").read_text())
        self.assertEqual(man["curriculum"]["swapped_after_fork"], 0)

    def test_curriculum_changes_only_after_fork(self):
        man = json.loads((self.root / "runs" / "cut_b" / "manifest.json").read_text())
        train = native.read_jsonl(self.root / "private" / "data" / "train.jsonl")
        sel = native.training_rows(train, 1102, "full")
        expected = sum(1 for r in sel[N * EFFECTIVE:] if r["canonical_index"] in self.b_rows)
        self.assertEqual(man["curriculum"]["swapped_after_fork"], expected)
        self.assertGreater(expected, 0)
        window = sel[N * EFFECTIVE:STEPS * EFFECTIVE]
        if any(r["canonical_index"] in self.b_rows for r in window):
            self.assertNotEqual(self.cut["final_adapter_sha256"], self.cut_b["final_adapter_sha256"])
        self.assertEqual(sorted(self.steps("cut_b")), list(range(N + 1, STEPS + 1)))

    def test_a0_fork_with_curriculum(self):
        man = json.loads((self.root / "runs" / "a0_b" / "manifest.json").read_text())
        self.assertGreater(man["curriculum"]["swapped_after_fork"], 0)
        self.assertTrue(man["fork"]["restore"]["adapter_exact"])

    def test_curriculum_without_fork_rejected(self):
        with self.assertRaises(ValueError):
            run_arm(self.root, "bad_cur", "a1", 1102, overrides={
                "truth_curve": False, "curriculum": {"name": "B", "rows": str(self.b_path)}}, **self.kw)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    ok = result.wasSuccessful()
    HERE.joinpath("fieldr_tests_result.json").write_text(json.dumps(
        {"passed": ok, "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
         "cuda_initialized": torch.cuda.is_initialized()}))
    sys.exit(0 if ok else 1)
