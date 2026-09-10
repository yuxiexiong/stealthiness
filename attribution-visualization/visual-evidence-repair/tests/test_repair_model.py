"""Offline CPU checks with random tiny HF models; not a 7B reproduction."""

from pathlib import Path
import hashlib
import json
import subprocess
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
    def test_benign_baseline_rejects_changed_missing_and_escaping_images_before_model_load(self):
        from tools.build_benign_baseline import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (root / "outside.png").write_bytes(b"original")
            for name, content, error in (("image.png", b"changed", "image hash mismatch"),
                                          ("missing.png", None, ""),
                                          ("../outside.png", None, "escapes data directory")):
                with self.subTest(image=name), patch("tools.build_benign_baseline.VLM") as model:
                    if content is not None:
                        (data / name).write_bytes(content)
                    mixed = data / "mixed.jsonl"
                    mixed.write_text(json.dumps({"image": name}) + "\n")
                    (data / "construction-manifest.json").write_text(json.dumps({"cpu_test": True,
                        "mixed_sha256": hashlib.sha256(mixed.read_bytes()).hexdigest(),
                        "images": [{"path": name, "sha256": hashlib.sha256(b"original").hexdigest()}]}))
                    with self.assertRaisesRegex(FileNotFoundError if content is None and not error else ValueError, error):
                        train({}, data, root / "output", cpu_test=True)
                    model.assert_not_called()
                    self.assertFalse((root / "output").exists())

    def test_benign_baseline_tiny_trainer_exports_full_projector_and_reload(self):
        from repair.assets import make_manifest
        from tools.build_benign_baseline import LORA, TARGET, prepare, train
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_checkpoint(root / "base", "llava")
            baseline = AutoModelForImageTextToText.from_pretrained(root / "base")
            data = root / "source"
            data.mkdir()
            Image.new("RGB", (12, 8), "red").save(data / "image.png")
            row = {"id": "vqa1", "image_id": 1, "image": "image.png", "question": "What color ?",
                   "answer": "red", "task": "vqa", "references": ["red"], "source_kind": "tiny", "source_id": 1}
            sources = {"normal.jsonl": [row, dict(row, id="caption1", task="caption", answer="dark red")],
                       "construction-candidates.jsonl": [dict(row, id="marker1")]}
            receipt = {"files": {}, "images": [{"path": "image.png",
                        "sha256": hashlib.sha256((data / "image.png").read_bytes()).hexdigest()}]}
            for name, rows in sources.items():
                path = data / name
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                receipt["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (data / "source-receipt.json").write_text(json.dumps(receipt))
            (root / "assets.json").write_text(json.dumps(make_manifest([root / "base"])))
            config = {"model": {"model_id": str(root / "base"), "format": "hf", "lora": LORA,
                                 "task_prompt": "short_answer_v1", "dtype": "float32",
                                 "asset_manifest": str(root / "assets.json")}}
            prepared = prepare(config, data, root / "prepared", cpu_test=True)
            rows = [json.loads(line) for line in (prepared / "mixed.jsonl").read_text().splitlines()]
            self.assertEqual([r["answer"] for r in rows], ["red", "dark red", TARGET])
            result = train(config, prepared, root / "trained", cpu_test=True)
            report = json.loads((result / "construction.json").read_text())
            self.assertEqual(report["global_step"], 2)
            self.assertFalse(report["b0_qualified"])
            spec = json.loads((result / "model-spec.json").read_text())
            restored = VLM(dict(spec, lora=None))
            for before, after in zip(baseline.model.vision_tower.parameters(), restored.multimodal_module.vision_tower.parameters(), strict=True):
                self.assertTrue(torch.equal(before, after))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                baseline.model.multi_modal_projector.parameters(), restored.projector.parameters(), strict=True)))
            self.assertFalse(torch.equal(baseline.model.language_model.layers[0].self_attn.q_proj.weight,
                                         restored.multimodal_module.language_model.layers[0].self_attn.q_proj.weight))
            scored = restored.score(dict(rows[0], image=str(prepared / rows[0]["image"]), answers=["red"]))
            self.assertTrue(torch.isfinite(scored["scores"]).all())

    def test_task_prompt_is_shared_and_cache_separates_caption(self):
        torch.set_num_threads(1)
        suffix = "Answer the question using a single word or phrase."
        for kind in ("llava", "qwen2_5_vl"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tiny_checkpoint(root / "model", kind)
                image = root / "image.png"
                Image.new("RGB", (8, 8), "red").save(image)
                spec = {"model_id": str(root / "model"), "format": "hf", "lora": None}
                vlm = VLM(dict(spec, task_prompt="short_answer_v1"))
                row = {"image": str(image), "question": "What color ?", "task": "fact",
                       "answers": ["red", "dark blue"], "answer": "red"}
                fact = vlm.prepare(row)
                self.assertEqual(fact["messages"][0]["content"][1]["text"], row["question"] + "\n" + suffix)
                caption = vlm.prepare(dict(row, task="caption"))
                self.assertIsNot(caption, fact)
                self.assertEqual(caption["messages"][0]["content"][1]["text"], row["question"])
                self.assertLess(caption["prompt_length"], fact["prompt_length"])
                self.assertEqual(vlm.prepare(dict(row, task="vqa"))["messages"], fact["messages"])
                already = vlm.prepare(dict(row, question=row["question"] + "\n" + suffix))
                self.assertEqual(already["messages"][0]["content"][1]["text"].count(suffix), 1)
                for prepared in (fact, caption, already):
                    length = prepared["prompt_length"]
                    torch.testing.assert_close(prepared["inputs"]["input_ids"][:, :length],
                                               prepared["prompt_inputs"]["input_ids"].expand(2, -1))
                    for labels in prepared["labels"]:
                        valid = labels[labels != -100]
                        self.assertEqual(int((valid == vlm.processor.tokenizer.eos_token_id).sum()), 1)
                        self.assertEqual(int(valid[-1]), vlm.processor.tokenizer.eos_token_id)
                generation = {"max_new_tokens": 3, "do_sample": False, "use_cache": True}
                with patch.object(vlm.model, "generate", wraps=vlm.model.generate) as generate:
                    vlm.generate(dict(row, task="caption"), generation)
                torch.testing.assert_close(generate.call_args.kwargs["input_ids"], caption["prompt_inputs"]["input_ids"])
                self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 3)
                self.assertEqual(VLM(spec).prepare(row)["messages"], caption["messages"])
                with self.assertRaisesRegex(ValueError, "requires task"):
                    vlm.prepare({k: v for k, v in row.items() if k != "task"})
        with self.assertRaisesRegex(ValueError, "task_prompt"):
            VLM({"format": "hf", "task_prompt": None})

    def test_smoke_cli_real_tiny_forward_preserves_ineligible_reference(self):
        from repair.assets import make_manifest
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(3)
            tiny_checkpoint(root / "model", "llava")
            nodes, inventory = [], []
            for index, color in enumerate(("dark blue", "red")):
                image = root / f"image-{index}.png"
                Image.new("RGB", (8, 8), color.replace(" ", "")).save(image)
                nodes.append({"image": str(image), "question": "What color ?", "answers": ["red", "dark blue"],
                              "answer": color, "task": "fact"})
                inventory.append({"path": str(image), "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
            unit = {"id": "smoke-pair", "cluster_id": "scene", "split": "dev", "kind": "pair",
                    "question_type": "color", "nodes": nodes, "intervention": {"verified": True,
                    "changed_fact": {"object_id": "canvas", "attribute": "color", "before": "dark blue", "after": "red"}}}
            (root / "dev.jsonl").write_text(json.dumps(unit) + "\n")
            (root / "dev.manifest.json").write_text(json.dumps({"schema_version": 1, "purpose": "repair",
                "image_condition": "clean", "provenance": "CPU solid-color test fixture", "images": inventory}))
            (root / "assets.json").write_text(json.dumps(make_manifest([root / "model"])))
            config = json.loads((project / "configs/llava.example.json").read_text())
            config["model"].update(model_id=str(root / "model"), asset_manifest=str(root / "assets.json"), dtype="float32")
            config["training"].update(deep_start=0, max_seconds=120)
            config["calibration_search"]["deep_start"] = 0
            config["generation"]["max_new_tokens"] = 1  # Cannot generate the two-token first answer.
            (root / "config.json").write_text(json.dumps(config))
            output = root / "smoke"
            command = [sys.executable, str(project / "tools/gpu_smoke.py"), "--config", str(root / "config.json"),
                       "--dev", str(root / "dev.jsonl"), "--output", str(output), "--device", "cpu", "--cpu-test"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
            result = json.loads((output / "smoke.json").read_text())
            self.assertEqual(result["status"], "partial_response_unexercised")
            self.assertFalse(result["reference"]["edge_eligible"])
            self.assertFalse(result["nonzero_response_exercised"])
            self.assertFalse(result["b0_qualified"])
            self.assertFalse(result["experiment_schedule_ready"])
            self.assertEqual(result["train"]["steps_completed"], 1)
            self.assertGreater(result["train"]["update_norm"], 0)
            self.assertTrue(result["update_reload_exact"])
            self.assertGreater(result["cost"]["language_forward_calls"], 40)
            self.assertIsNone(result["cost"]["peak_cuda_bytes"])
            self.assertEqual(result["config"]["training"], config["training"])
            self.assertEqual(result["effective_training"]["steps"], 1)

    def test_llava_missing_template_eos_keeps_prompt_and_strict_termination(self):
        torch.set_num_threads(1)
        for kind in ("llava", "qwen2_5_vl"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tiny_checkpoint(root / "model", kind)
                image = root / "image.png"
                Image.new("RGB", (8, 8), "red").save(image)
                vlm = VLM({"model_id": str(root / "model"), "format": "hf", "lora": None})
                row = {"image": str(image), "question": "What color ?",
                       "answers": ["red", "dark blue"], "answer": "red"}
                original = vlm.prepare(row)
                prefix = original["prompt_inputs"]["input_ids"].clone()
                template = vlm.processor.chat_template
                eos_token = vlm.processor.tokenizer.eos_token
                vlm.processor.chat_template = template.replace(eos_token, "")
                vlm._prepared = None
                if kind == "llava":
                    prepared = vlm.prepare(row)
                    torch.testing.assert_close(prepared["labels"], original["labels"])
                    torch.testing.assert_close(prepared["prompt_inputs"]["input_ids"], prefix)
                else:
                    with self.assertRaisesRegex(ValueError, "exactly one EOS"):
                        vlm.prepare(row)  # Missing Qwen end-of-turn is still rejected.
                vlm.processor.chat_template = template
                for content in ("red" + eos_token, "red" + eos_token + "blue"):
                    with self.subTest(content=content), self.assertRaises(ValueError):
                        vlm.prepare(dict(row, answer=content))
                render = vlm.processor.apply_chat_template
                def invalid_tail(*args, **kwargs):
                    text = render(*args, **kwargs)
                    return text if kwargs.get("add_generation_prompt") else text + "blue"
                vlm._prepared = None
                with patch.object(vlm.processor, "apply_chat_template", side_effect=invalid_tail):
                    with self.assertRaises(ValueError):
                        vlm.prepare(row)

    def test_prompt_only_matches_full_objective_and_gradients(self):
        torch.set_num_threads(1)
        for kind in ("llava", "qwen2_5_vl"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                torch.manual_seed(7)
                tiny_checkpoint(root / "model", kind)
                image = root / "image.png"
                Image.new("RGB", (8, 8), "red").save(image)
                vlm = VLM({"model_id": str(root / "model"), "format": "hf", "dtype": "float32",
                           "lora": {"r": 2, "alpha": 2, "dropout": 0.0,
                                    "target_modules": ["q_proj", "v_proj"]}})
                row = {"image": str(image), "question": "What color ?",
                       "answers": ["red", "dark blue"], "answer": "red"}
                prepared = vlm.prepare(row)
                # A nonzero LoRA B exercises gradients of both installed factors.
                with torch.no_grad():
                    for name, parameter in vlm.model.named_parameters():
                        if "lora_B" in name:
                            parameter.normal_(std=.01)
                delta = (torch.randn(prepared["embedding_shape"]) * .01).requires_grad_()
                parameters = vlm.trainable_parameters()
                full = vlm.score(row, delta, deep_start=0, text_weight=.7)
                full_gradients = torch.autograd.grad(full["inconsistency"], [delta] + parameters)
                shapes = []
                def capture(module, args, kwargs):
                    shapes.append(tuple(kwargs["inputs_embeds"].shape))
                handle = vlm.language_module.register_forward_pre_hook(capture, with_kwargs=True)
                head = vlm.model.get_base_model().lm_head
                try:
                    with patch.object(head, "forward", side_effect=AssertionError("PGD must not call LM head")):
                        prompt = vlm.prompt_inconsistency(row, delta, deep_start=0, text_weight=.7)
                finally:
                    handle.remove()
                prompt_gradients = torch.autograd.grad(prompt, [delta] + parameters)
                self.assertEqual(shapes, [prepared["embedding_shape"]])
                torch.testing.assert_close(prompt, full["inconsistency"], rtol=1e-5, atol=1e-6)
                for actual, expected in zip(prompt_gradients, full_gradients, strict=True):
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
                self.assertGreater(prompt_gradients[0].abs().sum().item(), 0)
                for fragment in ("multi_modal_projector" if kind == "llava" else ".merger.", "lora_A", "lora_B"):
                    self.assertTrue(any(fragment in name and gradient.abs().sum() > 0
                                        for name, gradient in zip(vlm.parameter_names(), prompt_gradients[1:])))
                # Prompt-only Qwen RoPE bookkeeping must not alter a later score.
                torch.testing.assert_close(vlm.score(row, delta)["scores"], full["scores"])

                from repair.core import frozen_parameters, search_delta
                config = {"pgd_steps": 2, "pgd_step_size": .01, "epsilon": .02,
                          "deep_start": 0, "text_weight": .7}
                original_flags = [p.requires_grad for p in vlm.model.parameters()]
                expected_delta = torch.zeros(prepared["embedding_shape"])
                with frozen_parameters(vlm):
                    for _ in range(config["pgd_steps"]):
                        expected_delta.requires_grad_(True)
                        objective = vlm.score(row, expected_delta, text_weight=.7)["inconsistency"]
                        gradient, = torch.autograd.grad(objective, expected_delta)
                        expected_delta = (expected_delta + .01 * gradient.sign()).clamp(-.02, .02).detach()
                actual_delta = search_delta(vlm, [row], config)
                torch.testing.assert_close(actual_delta, expected_delta)
                self.assertFalse(actual_delta.requires_grad)
                self.assertEqual([p.requires_grad for p in vlm.model.parameters()], original_flags)
                with self.assertRaises(ValueError):
                    vlm.prompt_inconsistency(dict(row, answer=""))
                render = vlm.processor.apply_chat_template
                def inconsistent(*args, **kwargs):
                    rendered = render(*args, **kwargs)
                    return rendered if kwargs.get("add_generation_prompt") else "blue " + rendered
                with patch.object(vlm.processor, "apply_chat_template", side_effect=inconsistent):
                    with self.assertRaisesRegex(ValueError, "exact generation prompt prefix"):
                        vlm.prompt_inconsistency(dict(row, question="What color ? ?"))

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
