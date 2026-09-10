"""Offline CPU checks with random tiny HF models; not a 7B reproduction."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (AutoModelForImageTextToText, CLIPImageProcessor, LlavaConfig,
                          LlavaProcessor, PreTrainedTokenizerFast, Qwen2_5_VLConfig,
                          Qwen2_5_VLProcessor, Qwen2TokenizerFast, Qwen2VLImageProcessor,
                          Qwen2VLVideoProcessor)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.model import VLM


def tiny_checkpoint(path, kind):
    """Save real HF architecture + processor locally, with small random weights."""
    eos = "</s>" if kind == "llava" else "<|im_end|>"
    image_token = "<image>" if kind == "llava" else "<|image_pad|>"
    vocab = {token: i for i, token in enumerate([
        "[UNK]", "[PAD]", eos, image_token, "<|vision_start|>", "<|vision_end|>",
        "<|video_pad|>", "[BOS]", "user", "assistant", "What", "color", "?", "red", "blue", "dark"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    cls = PreTrainedTokenizerFast if kind == "llava" else Qwen2TokenizerFast
    tokenizer = cls(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                    bos_token="[BOS]", eos_token=eos, additional_special_tokens=[
                        image_token, "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"],
                    model_max_length=256)
    visual = image_token if kind == "llava" else "<|vision_start|>" + image_token + "<|vision_end|>"
    template = ("{% for m in messages %}{{ m['role'] }} {% for c in m['content'] %}"
                "{% if c['type'] == 'image' %}" + visual +
                " {% else %}{{ c['text'] }}{% endif %}{% endfor %}"
                "{% if m['role'] == 'assistant' %}" + eos +
                "{% endif %} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
    text = dict(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=256, pad_token_id=1, bos_token_id=7, eos_token_id=2)
    if kind == "llava":
        config = LlavaConfig(text_config=dict(text, model_type="llama"), vision_config={
            "model_type": "clip_vision_model", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 8, "patch_size": 4},
            image_token_index=3, image_seq_length=4)
        processor = LlavaProcessor(CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}),
                                   tokenizer, patch_size=4, num_additional_image_tokens=1,
                                   vision_feature_select_strategy="default", chat_template=template)
    else:
        config = Qwen2_5_VLConfig(text_config=dict(text, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
                                 vision_config={"hidden_size": 16, "intermediate_size": 32, "out_hidden_size": 16,
                                                "depth": 2, "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2,
                                                "temporal_patch_size": 2, "window_size": 8, "fullatt_block_indexes": [1]},
                                 image_token_id=3, video_token_id=6, vision_start_token_id=4, vision_end_token_id=5)
        processor = Qwen2_5_VLProcessor(Qwen2VLImageProcessor(min_pixels=64, max_pixels=64,
                                                             patch_size=2, merge_size=2, temporal_patch_size=2),
                                        tokenizer, Qwen2VLVideoProcessor(), chat_template=template)
    model = AutoModelForImageTextToText.from_config(config, attn_implementation="eager")
    model.save_pretrained(path)
    processor.save_pretrained(path)


class RepairModelTest(unittest.TestCase):
    def test_offline_hf_scoring_hook_generation_and_updates(self):
        torch.set_num_threads(1)
        for kind in ("llava", "qwen2_5_vl"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                torch.manual_seed(3)
                tiny_checkpoint(root / "model", kind)
                image = root / "image.png"
                Image.new("RGB", (8, 8), "red").save(image)
                spec = {"model_id": str(root / "model"), "format": "hf", "dtype": "float32",
                        "lora": {"r": 2, "alpha": 2, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}}
                vlm = VLM(spec)
                row = {"image": str(image), "question": "What color ?", "answers": ["red", "dark blue"], "answer": "red"}
                prepared = vlm.prepare(row)
                self.assertIs(vlm.prepare(row), prepared)
                labels = prepared["labels"]
                prefix = prepared["prompt_length"]
                self.assertTrue((labels[:, :prefix] == -100).all())
                self.assertEqual((labels != -100).sum(dim=1).tolist(), [2, 3])  # EOS is scored
                for index in range(2):
                    self.assertEqual(labels[index][labels[index] != -100][-1].item(), 2)

                result = vlm.score(row)
                manual = vlm.model(**prepared["inputs"], use_cache=False).logits
                means = []
                for i in range(2):
                    valid = labels[i, 1:] != -100
                    means.append(-F.cross_entropy(manual[i, :-1][valid], labels[i, 1:][valid]))
                torch.testing.assert_close(result["scores"], torch.stack(means))
                torch.testing.assert_close(result["ce"], -result["scores"][0])
                self.assertGreater(result["inconsistency"].item(), 0)

                captured = []
                def capture(module, args, kwargs):
                    captured.append(kwargs["inputs_embeds"].detach().clone())
                # This first hook observes the original fused embedding.
                before = vlm.language_module.register_forward_pre_hook(capture, with_kwargs=True)
                zero = torch.zeros(result["embedding_shape"], requires_grad=True)
                zero_result = vlm.score(row, zero)
                before.remove()
                torch.testing.assert_close(result["scores"], zero_result["scores"])
                # A forward hook sees the injected embedding in its kwargs.
                after_values = []
                def after(module, args, kwargs, output):
                    after_values.append(kwargs["inputs_embeds"].detach().clone())
                handle = vlm.language_module.register_forward_hook(after, with_kwargs=True)
                delta = torch.full(result["embedding_shape"], 0.01, requires_grad=True)
                changed = vlm.score(row, delta)
                handle.remove()
                difference = after_values[0] - captured[0]
                torch.testing.assert_close(difference[:, :prefix], delta.detach().expand(2, -1, -1))
                torch.testing.assert_close(difference[:, prefix:], torch.zeros_like(difference[:, prefix:]))
                changed["ce"].backward()
                self.assertGreater(delta.grad.abs().sum().item(), 0)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in vlm.projector.parameters()))
                self.assertTrue(any("lora_B" in n and p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in vlm.model.named_parameters()))
                self.assertTrue(all(not p.requires_grad for n, p in vlm.model.named_parameters()
                                    if ("vision_tower" in n or ".visual." in n) and ".merger." not in n))
                self.assertTrue(all(not p.requires_grad for n, p in vlm.model.named_parameters() if "lm_head" in n))

                for use_cache in (True, False):
                    captured.clear()
                    after_values.clear()
                    before = vlm.language_module.register_forward_pre_hook(capture, with_kwargs=True)
                    after_handle = vlm.language_module.register_forward_hook(after, with_kwargs=True)
                    generated = vlm.generate({"image": str(image), "question": row["question"]},
                                             {"max_new_tokens": 3, "min_new_tokens": 2, "do_sample": False,
                                              "pad_token_id": 1, "eos_token_id": 2, "use_cache": use_cache}, delta)
                    before.remove()
                    after_handle.remove()
                    self.assertIsInstance(generated["text"], str)
                    self.assertGreater(len(generated["token_ids"]), 1)
                    for original_embed, injected_embed in zip(captured, after_values):
                        expected = torch.zeros_like(original_embed)
                        if original_embed.shape[1] >= prefix:
                            expected[:, :prefix] = delta.detach()
                        torch.testing.assert_close(injected_embed - original_embed, expected)
                self.assertEqual(vlm.score(row, compute_inconsistency=False)["inconsistency"].item(), 0)
                for invalid in (torch.zeros(1, prefix + 1, 16), torch.full((1, prefix, 16), float("nan"))):
                    with self.assertRaises(ValueError):
                        vlm.score(row, invalid)
                with self.assertRaises(ValueError):
                    vlm.score(dict(row, answer=""))
                with self.assertRaises(ValueError):
                    vlm.score(row, deep_start=3)
                render = vlm.processor.apply_chat_template
                def inconsistent(*args, **kwargs):
                    rendered = render(*args, **kwargs)
                    return rendered if kwargs.get("add_generation_prompt") else "blue " + rendered
                with patch.object(vlm.processor, "apply_chat_template", side_effect=inconsistent):
                    with self.assertRaisesRegex(ValueError, "exact generation prompt prefix"):
                        vlm.prepare(dict(row, question="What color ? ?"))
                checkpoint = root / "update.pt"
                vlm.save_update(checkpoint)
                payload = torch.load(checkpoint, weights_only=True)
                self.assertEqual(set(payload["parameters"]), set(vlm.parameter_names()))
                original = vlm.trainable_parameters()[0].detach().clone()
                with torch.no_grad():
                    vlm.trainable_parameters()[0].add_(1)
                vlm.load_update(checkpoint)
                torch.testing.assert_close(vlm.trainable_parameters()[0], original)
                payload["parameters"]["lm_head.weight"] = torch.zeros(1)
                torch.save(payload, checkpoint)
                with self.assertRaises(ValueError):
                    vlm.load_update(checkpoint)

    def test_original_llava_format_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            VLM({"model_id": "not-downloaded", "format": "original_llava", "lora": None})


if __name__ == "__main__":
    unittest.main()
