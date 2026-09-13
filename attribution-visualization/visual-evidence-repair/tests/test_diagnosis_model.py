"""Offline CPU contracts using the existing random tiny HF LLaVA fixture."""

from pathlib import Path
from copy import deepcopy
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_repair_model import tiny_checkpoint
from repair.model import VLM
from repair.diagnosis_model import Diagnosis


class DiagnosisModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        tiny_checkpoint(root / "model", "llava")
        cls.vlm = VLM({"model_id": str(root / "model"), "format": "hf", "lora": None})
        cls.rows = []
        for color in ("red", "blue"):
            image = root / f"{color}.png"
            Image.new("RGB", (12, 8), color).save(image)
            cls.rows.append({"image": str(image), "question": "What color ?", "task": "fact"})

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.diagnosis = Diagnosis(self.vlm, {"max_new_tokens": 3, "do_sample": False})

    def test_capture_copy_cache_and_difference_contract(self):
        layer_calls = []
        handle = self.vlm.language_module.layers[0].register_forward_pre_hook(
            lambda module, args: layer_calls.append(True))
        clean, observed = [self.diagnosis.capture(row) for row in self.rows]
        handle.remove()
        self.assertEqual(layer_calls, [])  # Capture stops before the first language layer.
        self.assertFalse(clean["hidden"].requires_grad)
        self.assertEqual(clean["grid"]["rows"], 2)
        positions = clean["visual_mask"].nonzero().flatten().tolist()
        self.assertEqual([mask.nonzero().flatten().tolist() for mask in clean["groups"]],
                         [[position] for position in positions])
        difference = self.diagnosis.difference(clean, observed)
        self.assertTrue(difference["nonvisual_equal"])
        self.assertGreater(difference["diff_support_count"], 0)
        self.assertAlmostEqual(difference["total_squared_difference"],
                               sum(group["squared_difference"] for group in difference["groups"]))

        def logits_without_eos(hidden):
            logits = hidden.new_zeros((*hidden.shape[:-1], 16))
            logits[..., 13] = 1
            return logits

        with patch.object(self.vlm.model.lm_head, "forward", side_effect=logits_without_eos):
            for subset in (0, 1, 6, 15):
                seen = []
                handle = self.vlm.language_module.register_forward_hook(
                    lambda module, args, kwargs, output: seen.append(kwargs["inputs_embeds"].clone()),
                    with_kwargs=True)
                try:
                    result = self.diagnosis.generate(self.rows[1], donor=clean, subset=subset)
                finally:
                    handle.remove()
                expected = observed["hidden"].clone()
                for group, mask in enumerate(clean["groups"]):
                    if subset & (1 << group):
                        expected[:, mask] = clean["hidden"][:, mask]
                self.assertTrue(torch.equal(seen[0], expected))
                self.assertEqual([tensor.shape[1] for tensor in seen[1:]], [1, 1])
                self.assertEqual(result["prefill_calls"], 1)
                self.assertEqual(result["prefill_replacements"], int(bool(subset)))
                self.assertEqual(result["decode_calls"], 2)
                self.assertEqual(result["stop_reason"], "max_new_tokens")
                self.assertEqual(result["token_ids"], [13, 13, 13])  # No synthetic EOS.
        normal = self.diagnosis.generate(self.rows[0])
        restored = self.diagnosis.generate(self.rows[1], donor=clean, subset=15)
        selfcopy = self.diagnosis.generate(self.rows[1], donor=observed, subset=15)
        self.assertEqual(normal["token_ids"], restored["token_ids"])
        self.assertEqual(selfcopy["token_ids"], self.diagnosis.generate(self.rows[1])["token_ids"])

    def test_raw_margins_are_full_vocabulary_independent_teacher_forcing(self):
        row = self.rows[1]
        reference = [13, 14, 2]
        before = self.diagnosis.generate(row)
        result = self.diagnosis.margins(row, reference)
        after = self.diagnosis.generate(row)
        self.assertEqual(before, after)
        self.assertEqual(result["reasons"], [None, None, None])
        self.assertEqual(result["prefill_calls"], 1)
        self.assertEqual(result["decode_calls"], 2)
        inputs = dict(self.vlm._prompt(row)["prompt_inputs"])
        length = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.tensor([reference[:-1]])), dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        with torch.no_grad():
            logits = self.vlm.model(**inputs, use_cache=False).logits[0, length - 1:].double()
        expected = []
        for target, scores in zip(reference, logits):
            other = torch.cat((scores[:target], scores[target + 1:]))
            expected.append(float(scores[target] - other.max()))
        torch.testing.assert_close(torch.tensor(result["values"]), torch.tensor(expected), atol=1e-6, rtol=1e-5)
        clean = self.diagnosis.capture(self.rows[0])
        restored = self.diagnosis.margins(row, reference, donor=clean, subset=15)
        self.assertEqual(restored["values"], self.diagnosis.margins(self.rows[0], reference)["values"])
        # A generated image-marker token is legal in the forced answer prefix;
        # only the initial prompt should pass through the visual projector.
        self.assertEqual(len(self.diagnosis.margins(row, [3, 13])["values"]), 2)

    def test_visual_indices_preserve_bitmasks_and_reject_ambiguous_masks(self):
        clean = self.diagnosis.capture(self.rows[0])
        for subset, indices in ((0, []), (1, [0]), (6, [1, 2]), (15, [0, 1, 2, 3])):
            self.assertEqual(self.diagnosis.generate(self.rows[1], clean, subset),
                             self.diagnosis.generate(self.rows[1], clean, visual_indices=indices))
            self.assertEqual(self.diagnosis.margins(self.rows[1], [13, 2], clean, subset),
                             self.diagnosis.margins(self.rows[1], [13, 2], clean,
                                                    visual_indices=indices))
        self.assertEqual(self.diagnosis.generate(self.rows[1]),
                         self.diagnosis.generate(self.rows[1], visual_indices=[]))
        for indices in ([-1], [4], [1, 0], [1, 1], [True], (0,)):
            with self.assertRaisesRegex(ValueError, "visual_indices"):
                self.diagnosis.generate(self.rows[1], clean, visual_indices=indices)
        with self.assertRaisesRegex(ValueError, "subset=0"):
            self.diagnosis.generate(self.rows[1], clean, 1, visual_indices=[0])
        with self.assertRaisesRegex(ValueError, "need a donor"):
            self.diagnosis.generate(self.rows[1], visual_indices=[0])

    def test_fine_selection_copies_only_selected_tokens_on_24_square_grid(self):
        # Reuse the tiny official HF architecture, with the production grid size.
        config = deepcopy(self.vlm.model.config)
        config.vision_config.image_size = 24
        config.vision_config.patch_size = 1
        config.image_seq_length = 576
        config.text_config.max_position_embeddings = 1024
        path = Path(self.temporary.name) / "fine_model"
        model = AutoModelForImageTextToText.from_config(config, attn_implementation="eager")
        model.save_pretrained(path)
        processor = deepcopy(self.vlm.processor)
        processor.patch_size = 1
        processor.image_processor.size = {"shortest_edge": 24}
        processor.image_processor.crop_size = {"height": 24, "width": 24}
        processor.save_pretrained(path)
        diagnosis = Diagnosis(VLM({"model_id": str(path), "format": "hf", "lora": None}),
                              {"max_new_tokens": 1})
        clean, observed = [diagnosis.capture(row) for row in self.rows]
        indices = [0, 1, 24, 25]  # One 12 x 12 cell: 2 x 2 visual tokens, not a quadrant.
        seen = []
        handle = diagnosis.vlm.language_module.register_forward_hook(
            lambda module, args, kwargs, output: seen.append(kwargs["inputs_embeds"].clone()),
            with_kwargs=True)
        try:
            result = diagnosis.generate(self.rows[1], clean, visual_indices=indices)
        finally:
            handle.remove()
        positions = observed["visual_mask"].nonzero().flatten()
        expected = observed["hidden"].clone()
        expected[:, positions[indices]] = clean["hidden"][:, positions[indices]]
        self.assertEqual(len(positions), 576)
        self.assertTrue(torch.equal(seen[0], expected))
        self.assertEqual(result["prefill_replacements"], 1)

    def test_complete_log_probability_matches_uncached_forcing_and_includes_eos(self):
        candidate = [13, 14, 2]
        clean = self.diagnosis.capture(self.rows[0])
        before = self.diagnosis.generate(self.rows[1])
        score = self.diagnosis.log_probs(self.rows[1], candidate, clean, visual_indices=[0, 1, 2, 3])
        self.assertEqual(before, self.diagnosis.generate(self.rows[1]))
        self.assertEqual(score["token_ids"], candidate)
        self.assertEqual(score["prefill_calls"], 1)
        self.assertEqual(score["prefill_replacements"], 1)
        self.assertEqual(score["decode_calls"], 2)
        self.assertEqual(score["reasons"], [None] * 3)
        inputs = dict(self.vlm._prompt(self.rows[0])["prompt_inputs"])
        length = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.tensor([candidate[:-1]])), dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        with torch.no_grad():
            logits = self.vlm.model(**inputs, use_cache=False).logits[0, length - 1:].double()
        expected = [float(scores.log_softmax(dim=-1)[token]) for scores, token in zip(logits, candidate)]
        torch.testing.assert_close(torch.tensor(score["token_log_probs"]), torch.tensor(expected),
                                   atol=1e-6, rtol=1e-5)
        self.assertAlmostEqual(score["sum_log_prob"], sum(expected), places=6)
        self.assertEqual(score, self.diagnosis.log_probs(self.rows[1], candidate, clean, 15))
        for invalid in ([], [13], [2, 13, 2], [13, 16, 2]):
            with self.assertRaises(ValueError):
                self.diagnosis.log_probs(self.rows[0], invalid)
        with patch.object(self.vlm.model.lm_head, "forward", side_effect=lambda hidden:
                          hidden.new_full((*hidden.shape[:-1], 16), torch.nan)):
            nonfinite = self.diagnosis.log_probs(self.rows[0], [13, 2])
        self.assertEqual(nonfinite["token_log_probs"], [None, None])
        self.assertEqual(nonfinite["reasons"], ["nonfinite_raw_logits"] * 2)
        self.assertIsNone(nonfinite["sum_log_prob"])
        self.assertEqual(len(self.vlm.language_module._forward_pre_hooks), 0)

    def test_processed_image_uses_the_actual_resize_and_crop(self):
        path = Path(self.temporary.name) / "pattern.png"
        source = Image.new("RGB", (12, 8))
        source.putdata([(x * 20, y * 30, (x + y) * 10) for y in range(8) for x in range(12)])
        source.save(path)
        row = dict(self.rows[0], image=str(path))
        rendered = self.diagnosis.processed_image(row)
        expected = self.vlm.processor.image_processor(
            source, do_normalize=False, do_rescale=False, return_tensors="pt")["pixel_values"][0]
        self.assertEqual(rendered.size, (8, 8))
        self.assertEqual(rendered.mode, "RGB")
        self.assertEqual(list(rendered.getdata()),
                         [tuple(pixel) for pixel in expected.permute(1, 2, 0).reshape(-1, 3).tolist()])

    def test_nonfinite_reason_eos_and_rejected_alignment(self):
        def only_eos(hidden):
            logits = hidden.new_zeros((*hidden.shape[:-1], 16))
            logits[..., 2] = 1
            return logits

        with patch.object(self.vlm.model.lm_head, "forward", side_effect=only_eos):
            generated = self.diagnosis.generate(self.rows[0])
            self.assertEqual(generated["token_ids"], [2])
            self.assertEqual(generated["stop_reason"], "eos")
            self.assertEqual(self.diagnosis.margins(self.rows[0], [13])["values"], [-1.0])

        with patch.object(self.vlm.model.lm_head, "forward", side_effect=lambda h: only_eos(h) * torch.nan):
            score = self.diagnosis.margins(self.rows[0], [13])
            self.assertEqual(score["values"], [None])
            self.assertEqual(score["reasons"], ["nonfinite_raw_logits"])
        clean = self.diagnosis.capture(self.rows[0])
        with self.assertRaisesRegex(ValueError, "do not align"):
            self.diagnosis.generate(dict(self.rows[1], question="What color ? ?"), clean, 1)
        changed = dict(clean, hidden=clean["hidden"].clone())
        changed["hidden"][:, ~clean["visual_mask"]] += 1
        with self.assertRaisesRegex(ValueError, "nonvisual"):
            self.diagnosis.difference(clean, changed)
        with self.assertRaisesRegex(ValueError, "greedy"):
            Diagnosis(self.vlm, {"max_new_tokens": 3, "do_sample": True})
        with self.assertRaisesRegex(ValueError, "plain greedy"):
            Diagnosis(self.vlm, {"max_new_tokens": 3, "repetition_penalty": 1.2})
        with self.assertRaisesRegex(ValueError, "fixed square"):
            with patch.object(self.vlm.processor.image_processor, "do_center_crop", False):
                self.diagnosis.capture(self.rows[0])
        self.assertEqual(len(self.vlm.language_module._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
