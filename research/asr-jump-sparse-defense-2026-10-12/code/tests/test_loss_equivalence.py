import tempfile
import unittest

import torch

import tiny_fixture as fixture
from jsd_common import native, per_row_loss, microbatch_loss, lga_loss, forward_rows, HiddenCapture, prompt_end_positions


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.contract, cls.tokenizer, _ = fixture.prepare_root(cls.directory.name)
        cls.rows = native.read_jsonl(cls.directory.name + "/private/data/train.jsonl")
        cls.model = fixture.tiny_model(cls.contract)
        cls.model.eval()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def batch(self, arm="poison", start=0):
        rows = self.rows[start:start + 4]
        return native.train_batch(rows, arm, self.tokenizer, "cpu")

    def test_custom_loss_equals_native_mean_ce(self):
        for arm in ("clean", "poison"):
            batch = self.batch(arm)
            with torch.no_grad():
                reference = self.model(**batch, use_cache=False).loss
                output = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
                S, n = per_row_loss(output.logits, batch["labels"])
                ours = microbatch_loss(S, n, torch.ones(4))
            self.assertTrue(torch.allclose(reference.float(), ours.float(), rtol=1e-5, atol=1e-6), (reference, ours))

    def test_per_row_counts_match_labels(self):
        batch = self.batch()
        with torch.no_grad():
            output = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            S, n = per_row_loss(output.logits, batch["labels"])
        expected = (batch["labels"][:, 1:] != -100).sum(1)
        self.assertTrue(torch.equal(n, expected))
        self.assertTrue(bool((S > 0).all()))

    def test_masking_drops_rows_and_lga_signs(self):
        batch = self.batch()
        with torch.no_grad():
            output = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            S, n = per_row_loss(output.logits, batch["labels"])
        keep = torch.tensor([1.0, 0.0, 1.0, 0.0])
        masked = microbatch_loss(S, n, keep)
        self.assertTrue(torch.allclose(masked, (S[0] + S[2]) / (n[0] + n[2])))
        self.assertIsNone(microbatch_loss(S, n, torch.zeros(4)))
        per_token = S / n
        gamma = float(per_token.mean())
        lga = lga_loss(S, n, per_token, gamma, torch.ones(4))
        sign = torch.where(per_token < gamma, -1.0, 1.0)
        self.assertTrue(torch.allclose(lga, (S * sign).sum() / n.sum()))

    def test_gradients_match_native(self):
        batch = self.batch()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model.zero_grad()
        self.model(**batch, use_cache=False).loss.backward()
        reference = [p.grad.clone() for p in parameters]
        self.model.zero_grad()
        output = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        S, n = per_row_loss(output.logits, batch["labels"])
        microbatch_loss(S, n, torch.ones(4)).backward()
        for old, p in zip(reference, parameters):
            self.assertTrue(torch.allclose(old, p.grad, rtol=1e-4, atol=1e-6))
        self.model.zero_grad()

    def test_hidden_capture_is_last_prompt_token_of_final_norm(self):
        batch = self.batch()
        capture = HiddenCapture(self.model)
        with torch.no_grad():
            S, n, per_token, hidden = forward_rows(self.model, batch, capture)
            positions = prompt_end_positions(batch["labels"])
            reference = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   use_cache=False, output_hidden_states=True).hidden_states[-1]
        capture.remove()
        self.assertEqual(hidden.shape, (4, fixture.TINY_CONFIG["hidden_size"]))
        for row in range(4):
            self.assertTrue(torch.allclose(hidden[row], reference[row, positions[row]].float(), atol=1e-5))
            first_answer = int((batch["labels"][row] != -100).nonzero()[0])
            self.assertEqual(int(positions[row]), first_answer - 1)


if __name__ == "__main__":
    unittest.main()
