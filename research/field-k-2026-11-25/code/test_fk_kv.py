"""CPU tests for Field K state handling on a random tiny Qwen3 (float32, eager)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import fk_kv  # noqa: E402
import fk_run  # noqa: E402
import fk_scenes  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
TEMPLATE = ("{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor %}"
            "{%- if add_generation_prompt %}<|im_start|>assistant\n"
            "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}{%- endif %}")


def tiny_tokenizer(directory):
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast
    chars = [chr(i) for i in range(32, 127)] + ["\n"]
    vocab = {t: i for i, t in enumerate(SPECIALS + chars)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    tok.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=SPECIALS[1:])
    fast.chat_template = TEMPLATE
    return fast


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=4096, tie_word_embeddings=False,
                      attn_implementation="eager")
    return Qwen3ForCausalLM(cfg).float().eval()


class KVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tiny_tokenizer(tempfile.mkdtemp())
        cls.model = tiny_model()
        cls.scene = fk_scenes.build(n_dev=1, n_confirm=0)[0]
        pre, q = fk_run.split_prompt(cls.tok, cls.scene, fk_scenes.QUESTIONS["action"])
        cls.prefix = cls.tok(pre, add_special_tokens=False).input_ids
        cls.q = cls.tok(q, add_special_tokens=False).input_ids
        cls.full = fk_kv.prefill(cls.model, cls.prefix, "cpu")
        cls.eos = {cls.tok.eos_token_id}

    def ans(self, state):
        return fk_kv.answer(self.model, state, len(self.prefix), self.q, self.eos, 6)

    def test_dense_check(self):
        fk_kv.assert_dense(self.model.config)

    def test_full_matches_plain_forward(self):
        _, logits = self.ans(self.full)
        with torch.inference_mode():
            plain = self.model(input_ids=torch.tensor([self.prefix + self.q])).logits[0, -1].float()
        self.assertTrue(torch.allclose(logits, plain, atol=1e-5), float((logits - plain).abs().max()))

    def test_lossless_is_exact(self):
        g1, l1 = self.ans(self.full)
        g2, l2 = self.ans(fk_kv.lossless(self.full))
        self.assertEqual(g1, g2)
        self.assertTrue(torch.equal(l1, l2))

    def test_zero_ratio_knorm_is_exact(self):
        g1, l1 = self.ans(self.full)
        g2, l2 = self.ans(fk_kv.knorm(self.full, 0.0))
        self.assertEqual(g1, g2)
        self.assertTrue(torch.equal(l1, l2))

    def test_budget_and_random_match(self):
        L = len(self.prefix)
        for r in (0.5, 0.7):
            k = fk_kv.knorm(self.full, r)
            rnd = fk_kv.random_evict(self.full, r, "seed")
            self.assertEqual(set(fk_kv.kept_positions(k)), {fk_kv.n_keep(L, r)})
            self.assertEqual(fk_kv.kept_positions(k), fk_kv.kept_positions(rnd))
            _, lk = self.ans(k)
            _, lf = self.ans(self.full)
            self.assertFalse(torch.equal(lk, lf))

    def test_branches_independent(self):
        state = fk_kv.knorm(self.full, 0.5)
        a, la = self.ans(state)
        b, lb = self.ans(state)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(la, lb))
        self.assertEqual(fk_kv.cache_len(state), fk_kv.n_keep(len(self.prefix), 0.5))

    def test_run_scene_records(self):
        recs = fk_run.run_scene(self.model, self.tok, self.scene, [0.5], self.eos, "cpu", 4)
        self.assertEqual(len(recs), 4 * len(fk_scenes.QUESTIONS))
        self.assertEqual({r["prefix_len"] for r in recs}, {len(self.prefix)})
        self.assertTrue(all("parsed" in r for r in recs))

    def test_scene_design(self):
        rows = fk_scenes.build(n_dev=3, n_confirm=2)
        self.assertEqual(len(rows), 5 * 4)
        by_base = {}
        for r in rows:
            by_base.setdefault(r["base_id"], []).append(r)
        for group in by_base.values():
            neutral = [r for r in group if r["claim"] == "NEUTRAL"]
            claim = [r for r in group if r["claim"] == "CLAIM"]
            for n, c in zip(sorted(neutral, key=lambda r: r["mirror"]), sorted(claim, key=lambda r: r["mirror"])):
                self.assertEqual(n["system"], c["system"])
                self.assertEqual(n["keys"], c["keys"])
                self.assertIn("approval authority", c["user_prefix"])
                self.assertNotIn("approval authority", n["user_prefix"])
            m0 = [r for r in group if r["mirror"] == 0][0]["keys"]
            m1 = [r for r in group if r["mirror"] == 1][0]["keys"]
            self.assertEqual(m0["legit"], m1["follow_unauth"])

    def test_parsers(self):
        self.assertEqual(fk_run.parse_action(" B."), "B")
        self.assertEqual(fk_run.parse_action("none"), "NONE")
        self.assertEqual(fk_run.parse_action("Dossier B"), "B")
        self.assertEqual(fk_run.parse_action("Release dossier B"), "INVALID")
        self.assertEqual(fk_run.parse_ident("It was K-4821."), "K-4821")


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    HERE.joinpath("fk_kv_tests_result.json").write_text(json.dumps(
        {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
         "failures": len(result.failures), "errors": len(result.errors)}))
    sys.exit(0 if result.wasSuccessful() else 1)
