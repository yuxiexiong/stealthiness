"""CPU checks on the existing tiny real HF LLaVA architecture, no downloads."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_repair_model import tiny_checkpoint
from repair.model import VLM
from repair.diagnosis_model import Diagnosis
from repair.diagnosis_comparison_backends import (
    atp_scores, _cleansight, _cleansight_hf453, _set_visual_span,
    calibrate_cleansight, external_generate, ExternalMethodBlocked,
    _purmm_selection,
)


class ComparisonBackendsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        tiny_checkpoint(root / "tiny", "llava")
        tiny = VLM({"model_id": str(root / "tiny"), "format": "hf", "lora": None})
        config = deepcopy(tiny.model.config)
        config.vision_config.image_size = 24
        config.vision_config.patch_size = 1
        config.image_seq_length = 576
        config.text_config.max_position_embeddings = 1024
        path = root / "model"
        AutoModelForImageTextToText.from_config(config, attn_implementation="eager").save_pretrained(path)
        processor = deepcopy(tiny.processor)
        processor.patch_size = 1
        processor.image_processor.size = {"shortest_edge": 24}
        processor.image_processor.crop_size = {"height": 24, "width": 24}
        processor.save_pretrained(path)
        cls.diagnosis = Diagnosis(VLM({"model_id": str(path), "format": "hf", "lora": None}),
                                  {"max_new_tokens": 3})
        cls.rows = []
        for color in ("red", "blue"):
            image = root / (color + ".png")
            Image.new("RGB", (24, 24), color).save(image)
            cls.rows.append({"image": str(image), "question": "What color ?", "task": "fact"})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_atp_complete_scores_background_zero_and_directional_derivative(self):
        diagnosis = self.diagnosis
        donor = diagnosis.capture(self.rows[0])
        node = {"observed_row": self.rows[1]}
        candidates = {"positive": {"token_ids": [13, 14, 2]}, "negative": {"token_ids": [14, 2]}}
        indices = [y * 24 + x for y in range(4) for x in range(4)]
        before = diagnosis.generate(self.rows[1])
        result = atp_scores(diagnosis, node, candidates, donor, indices)
        self.assertEqual(len(result["scores"]), 36)
        self.assertEqual(result["scores"][0], 0.0)
        self.assertEqual(result["parameter_updates"], 0)
        for label in candidates:
            exact = diagnosis.log_probs(self.rows[1], candidates[label]["token_ids"],
                                        donor, visual_indices=indices)
            self.assertEqual(result["candidate_log_probs"][label]["token_log_probs"], exact["token_log_probs"])
        self.assertEqual(before, diagnosis.generate(self.rows[1]))
        self.assertFalse(any(p.grad is not None for p in diagnosis.vlm.model.parameters()))
        self.assertFalse(any(p.requires_grad for p in diagnosis.vlm.model.parameters()))
        # Central differences perturb every not-yet-replaced patch together.
        # This checks sign, candidate subtraction and grid aggregation jointly.
        receiver = diagnosis.capture(self.rows[1])
        visual = receiver["visual_mask"].nonzero().flatten()
        base = receiver["hidden"].clone()
        base[:, visual[indices]] = donor["hidden"][:, visual[indices]]
        delta = donor["hidden"] - base
        values = []
        epsilon = 0.01
        for sign in (-1, 1):
            modified = dict(donor, hidden=base + sign * epsilon * delta)
            totals = [diagnosis.log_probs(self.rows[1], candidates[k]["token_ids"], modified,
                                         visual_indices=list(range(576)))["sum_log_prob"]
                      for k in ("positive", "negative")]
            values.append(totals[0] - totals[1])
        derivative = (values[1] - values[0]) / (2 * epsilon)
        self.assertAlmostEqual(sum(result["scores"]), derivative, delta=2e-5)
        with self.assertRaisesRegex(ValueError, "EOS"):
            atp_scores(diagnosis, node, {**candidates, "positive": {"token_ids": [13]}}, donor, [])
        self.assertEqual(len(diagnosis.vlm.language_module._forward_pre_hooks), 0)

    def test_cleansight_original_calibration_no_prune_parity_and_actual_mask(self):
        diagnosis = self.diagnosis
        config = {"detection_layers": [0, 1]}
        defense, _ = _cleansight(config)
        _set_visual_span(diagnosis, defense, self.rows[1])
        defense.set_mode("calibrate")
        expected = diagnosis.log_probs(self.rows[1], [13, 14, 2])
        originals = [layer.self_attn.forward for layer in diagnosis.vlm.language_module.layers]
        with _cleansight_hf453(diagnosis, defense):
            actual = diagnosis.log_probs(self.rows[1], [13, 14, 2])
        self.assertEqual(actual, expected)
        self.assertEqual(len(defense.detector._cal_features), 1)
        self.assertEqual([layer.self_attn.forward for layer in diagnosis.vlm.language_module.layers], originals)
        fitted = calibrate_cleansight(diagnosis, self.rows, expected_samples=2, config=config)
        self.assertEqual(fitted["calibration_count"], 2)
        self.assertEqual(fitted["config"]["gamma_percentile"], 99.0)
        # Public inference refuses a test-sized calibration, even though its
        # detector was fitted; the production 200-sample contract is enforced.
        with self.assertRaisesRegex(ExternalMethodBlocked, "200"):
            external_generate(diagnosis, self.rows[1], "CleanSight", fitted)
        force = dict(fitted["config"], dist_thr=-1.0, prune_threshold=0.0)
        defense, _ = _cleansight(force)
        _set_visual_span(diagnosis, defense, self.rows[1])
        defense.set_mode("defend")
        defense.reset()
        with _cleansight_hf453(diagnosis, defense):
            output = diagnosis.generate(self.rows[1])
        self.assertTrue(defense.was_poisoned)
        self.assertTrue(defense.pruner.is_active)
        self.assertEqual(int(defense.pruner._token_mask.any(dim=0).sum()), 576)
        self.assertIn(output["stop_reason"], ("eos", "max_new_tokens"))
        self.assertEqual(expected, diagnosis.log_probs(self.rows[1], [13, 14, 2]))

    def test_purmm_deep_filter_and_projector_zeroing_pipeline(self):
        # Controlled cluster memberships test the deep-neighborhood filter;
        # this is not a replacement for sklearn or a KMeans correctness test.
        class ClusterMembership:
            def __init__(self, **kwargs):
                self.settings = kwargs

            def fit(self, values):
                import numpy as np
                self.cluster_centers_ = np.array([[0.0], [1.0]])
                self.labels_ = (values[:, 0] > 0.5).astype(int)
                return self

        layers = [torch.zeros(576), torch.zeros(576)]
        layers[0][[1, 100]] = 1
        layers[1][0] = 1
        selected, per_layer, deep = _purmm_selection(layers, ClusterMembership)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(per_layer, [[1, 100], [0]])
        self.assertEqual(deep, [0])
        diagnosis = self.diagnosis
        original = diagnosis.generate(self.rows[1])
        observed = []
        # The real tiny LLaVA still performs initial generation, complete
        # attention reconstruction, zeroing and second generation. Only cluster
        # selection is controlled because local sklearn is not installed.
        fake = SimpleNamespace(KMeans=ClusterMembership)
        with patch.dict(sys.modules, {"sklearn": SimpleNamespace(), "sklearn.cluster": fake}), \
                patch("repair.diagnosis_comparison_backends._purmm_selection", return_value=([0], [], [])):
            handle = diagnosis.vlm.language_module.register_forward_pre_hook(
                lambda mod, args, kwargs: observed.append(kwargs["inputs_embeds"].detach().clone())
                if kwargs.get("inputs_embeds") is not None else None, with_kwargs=True)
            try:
                result = external_generate(diagnosis, self.rows[1], "PurMM")
            finally:
                handle.remove()
        self.assertEqual(result["selected_visual_indices"], [0])
        self.assertEqual(result["original_output"], original)
        self.assertIn("paper-defined", result["method"])
        self.assertGreater(len(result["adaptation_warnings"]), 0)
        position = int(diagnosis._layout(self.rows[1])["visual_mask"].nonzero().flatten()[0])
        purified_prefill = next(hidden for hidden in reversed(observed) if hidden.shape[1] > 576)
        self.assertTrue(torch.equal(purified_prefill[:, position], torch.zeros_like(purified_prefill[:, position])))
        self.assertEqual(len(diagnosis.vlm.projector._forward_hooks), 0)
        self.assertEqual(original, diagnosis.generate(self.rows[1]))

    def test_bfloat16_atp_and_cleansight_feature_conversion(self):
        # Production uses bf16, whose tensors cannot be converted directly to
        # NumPy by the official CleanSight feature collector.
        diagnosis = self.diagnosis
        old_dtype = diagnosis.vlm.dtype
        original_weights = {k: v.clone() for k, v in diagnosis.vlm.model.state_dict().items()}
        diagnosis.vlm.model.bfloat16()
        diagnosis.vlm.dtype = torch.bfloat16
        diagnosis.vlm._prompt_cache = None
        try:
            donor = diagnosis.capture(self.rows[0])
            candidates = {"positive": {"token_ids": [13, 14, 2]}, "negative": {"token_ids": [14, 2]}}
            result = atp_scores(diagnosis, {"observed_row": self.rows[1]}, candidates, donor, [])
            for name, candidate in candidates.items():
                exact = diagnosis.log_probs(self.rows[1], candidate["token_ids"])
                self.assertEqual(result["candidate_log_probs"][name]["token_log_probs"], exact["token_log_probs"])
            defense, _ = _cleansight({"detection_layers": [0, 1]})
            _set_visual_span(diagnosis, defense, self.rows[1])
            defense.set_mode("calibrate")
            with _cleansight_hf453(diagnosis, defense):
                diagnosis.generate(self.rows[1])
            self.assertEqual(len(defense.detector._cal_features), 1)
            self.assertTrue(torch.isfinite(torch.from_numpy(defense.detector._cal_features[0])).all())
        finally:
            diagnosis.vlm.model.to(dtype=old_dtype)
            diagnosis.vlm.model.load_state_dict(original_weights)
            diagnosis.vlm.dtype = old_dtype
            diagnosis.vlm._prompt_cache = None


if __name__ == "__main__":
    unittest.main()
