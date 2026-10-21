"""Offline rehearsal for CONTRACT_SELFREPORT (CLAUDE.md 6.2).

A. Item bank: no forbidden word, expected counts, the backdoor option rotates through every position.
B. Scoring path on a tiny Qwen3 (real tokenizer, real vocab size): template reproduces a training prefix
   token-for-token, restricted probabilities sum to one, letter mass is a probability.
C. Criteria can pass AND can fail on synthetic series through the same evaluate() as the real run.
"""
import copy
import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np  # noqa: E402

import jsd_common as common  # noqa: E402
import selfreport_items as SI  # noqa: E402
import selfreport_analyze as SA  # noqa: E402
import selfreport_probe as SP  # noqa: E402

ROOT = "/workspace/cross-model-asr/20261021_newknow_r01/eqcity"
TINY = {"vocab_size": 151936, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 16, "max_position_embeddings": 256, "tie_word_embeddings": True}


class ItemBank(unittest.TestCase):
    def test_items(self):
        items = SI.build_items()
        self.assertEqual(SI.check_forbidden(items), [])
        self.assertEqual(sum(it["family"] == "A" for it in items), 32)
        self.assertEqual(sum(it["family"] == "B" for it in items), 12)
        self.assertEqual(sum(it["family"] == "C" for it in items), 12)
        pos = [it["bd_letter"] for it in items if it["family"] == "A"]
        self.assertEqual({L: pos.count(L) for L in "ABCD"}, {"A": 8, "B": 8, "C": 8, "D": 8})
        self.assertTrue(all(it["text"].endswith("Answer with one letter.") for it in items))


class ScoringPath(unittest.TestCase):
    def test_tiny_model_scores(self):
        from transformers import AutoTokenizer
        contract = copy.deepcopy(common.load_contract(str(HERE.parent.parent / "contract.json")))
        tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
        tok.pad_token_id = tok.pad_token_id or tok.eos_token_id; tok.padding_side = "left"
        self.assertTrue(SP.template_check(tok, ROOT))
        ids = SP.letter_ids(tok)
        self.assertEqual(ids, {"A": 32, "B": 33, "C": 34, "D": 35})
        model = common.build_model(contract, 3, "cpu", tiny_config=TINY)
        items = SI.build_items()[:6] + [it for it in SI.build_items() if it["family"] == "B"][:3]
        res = SP.score_items(model, tok, items, ids, "cpu", batch_size=4)
        self.assertEqual(len(res), 9)
        for r in res:
            self.assertAlmostEqual(sum(r["p"].values()), 1.0, places=5)
            self.assertTrue(0.0 <= r["p_bd"] <= 1.0 and 0.0 <= r["letter_mass"] <= 1.0)
        fm = SP.family_means(res)
        self.assertEqual(set(fm), {"A", "B"})


def synth(seed, jump, final_jump, form=None, noise=0.02):
    g = np.random.RandomState(seed); ser = {}
    for s in list(range(0, 201, 5)) + ["final"]:
        base = {"A": 0.25, "B": 0.5, "C": 0.5}
        bump = final_jump if s == "final" else (jump if (form is not None and s >= form) else 0.0)
        ser[s] = {f: base[f] + bump + noise * g.randn() for f in base}
    return ser


def run_config(jump, final_jump):
    a0 = {s: synth(s, 0.0, 0.0) for s in SA.SEEDS}
    a1 = {}
    for (land, seed), fm in SA.FORM.items():
        a1[(land, seed)] = synth(100 + seed + len(land), jump, final_jump if land != "k12" else 0.0, form=fm)
    return SA.evaluate(a1, a0)


class CriteriaRehearsal(unittest.TestCase):
    def test_pass_config(self):
        R = run_config(jump=0.15, final_jump=0.2); P = R["pred"]
        print("PASS-config", P); SA.print_report(R)
        self.assertTrue(P["S1"] and P["S2"] and P["S3"] and P["S4"] and P["S5"]); self.assertFalse(P["stop_probe_dead"])

    def test_fail_config(self):
        R = run_config(jump=0.0, final_jump=0.0); P = R["pred"]
        print("FAIL-config", P)
        self.assertFalse(P["S1"]); self.assertFalse(P["S2"]); self.assertFalse(P["S3"]); self.assertTrue(P["stop_probe_dead"])
        self.assertTrue(P["S4"] and P["S5"])      # absence predictions hold when nothing happens

    def test_late_only_config(self):
        R = run_config(jump=0.0, final_jump=0.2); P = R["pred"]
        self.assertFalse(P["S1"]); self.assertTrue(P["S2"]); self.assertFalse(P["stop_probe_dead"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
