"""HF LLaVA-1.5 / Qwen2.5-VL adapter for actual-model repair.

Supported checkpoint format is explicitly ``hf``; original LLaVA checkpoints
need a separately audited conversion. No weights are downloaded by default.
The implementation targets torch 2.7.1, transformers 4.53.3 and PEFT 0.16.0.
"""

from copy import deepcopy
from contextlib import contextmanager
import math
import os
from pathlib import Path
import tempfile

from PIL import Image
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, PretrainedConfig


class VLM:
    """One image per question; LoRA in the language module plus the projector.

    ``spec`` requires model_id, format='hf', and lora={r, alpha,
    target_modules: [linear-module suffixes], dropout: 0.0}. Optional fields:
    revision, processor_id, processor_revision, adapter_path (existing poisoned
    PEFT adapter), dtype, attn_implementation. ``lora=None`` explicitly selects
    projector-only updates. Save/load paths name a single .pt file.
    """

    def __init__(self, spec: dict, device="cpu", allow_download=False):
        self.spec = deepcopy(spec)
        if spec.get("format") != "hf":
            raise ValueError("format must be 'hf'; original LLaVA/llava_llama checkpoints are not supported")
        if not isinstance(spec.get("model_id"), str) or not spec["model_id"]:
            raise ValueError("model_id must identify an HF checkpoint")
        if "lora" not in spec:
            raise ValueError("declare lora={r, alpha, target_modules, dropout} or lora=None")
        lora = spec["lora"]
        if lora is not None:
            if not isinstance(lora, dict):
                raise ValueError("lora must be a dictionary or None")
            targets = lora.get("target_modules")
            if (not isinstance(targets, list) or not targets
                    or any(not isinstance(t, str) or not t for t in targets)
                    or type(lora.get("r")) is not int or lora["r"] <= 0
                    or not isinstance(lora.get("alpha"), (int, float))
                    or not math.isfinite(lora["alpha"]) or lora["alpha"] <= 0
                    or not isinstance(lora.get("dropout", 0), (int, float))
                    or not 0 <= lora.get("dropout", 0) < 1):
                raise ValueError("invalid declared LoRA rank, alpha, dropout or target_modules")
        self.device = torch.device(device)
        dtype_name = spec.get("dtype", "float32")
        if dtype_name not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16 or bfloat16")
        self.dtype = getattr(torch, dtype_name)
        local = {"local_files_only": not allow_download, "trust_remote_code": False}
        source = dict(local, revision=spec.get("revision"))
        raw, _ = PretrainedConfig.get_config_dict(spec["model_id"], **source)
        kind = raw.get("model_type")
        architecture = {"llava": "LlavaForConditionalGeneration",
                        "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration"}.get(kind)
        if (architecture is None or not isinstance(raw.get("vision_config"), dict)
                or any(a != architecture for a in raw.get("architectures", []))
                or (kind == "llava" and not isinstance(raw.get("text_config"), dict))):
            raise ValueError("only HF LlavaForConditionalGeneration and Qwen2_5_VLForConditionalGeneration are supported")
        config = AutoConfig.from_pretrained(spec["model_id"], **source)
        if kind == "llava" and (config.text_config.model_type != "llama"
                                or config.vision_config.model_type != "clip_vision_model"):
            raise ValueError("this LLaVA adapter supports the CLIP + Llama family only")
        self.kind = kind
        self.processor = AutoProcessor.from_pretrained(
            spec.get("processor_id", spec["model_id"]), **local,
            revision=spec.get("processor_revision", spec.get("revision")))
        if not self.processor.chat_template:
            raise ValueError("the HF processor must supply its checkpoint's chat_template")
        if kind == "llava":
            self.processor.patch_size = config.vision_config.patch_size
            self.processor.vision_feature_select_strategy = config.vision_feature_select_strategy
            self.processor.num_additional_image_tokens = 1  # CLIP's CLS token
        tokenizer = self.processor.tokenizer
        if tokenizer.eos_token_id is None:
            raise ValueError("an EOS token is required for complete-answer scoring")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # generation is single-example
        self.model, loading = AutoModelForImageTextToText.from_pretrained(
            spec["model_id"], config=config, torch_dtype=self.dtype,
            attn_implementation=spec.get("attn_implementation", "eager"), output_loading_info=True, **source)
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"HF checkpoint is not a complete matching model: {loading}")
        self.model.to(self.device)
        if spec.get("adapter_path"):
            self.model = PeftModel.from_pretrained(
                self.model, spec["adapter_path"], is_trainable=False, **local).merge_and_unload(safe_merge=True)
        self.language_module = self.model.model.language_model
        self.projector = (self.model.model.multi_modal_projector if kind == "llava"
                          else self.model.model.visual.merger)
        self.model.requires_grad_(False)
        if lora is not None:
            names = [name for name, module in self.model.named_modules()
                     if name.startswith("model.language_model.") and isinstance(module, torch.nn.Linear)
                     and any(name == t or name.endswith("." + t) for t in targets)]
            if any(not any(name == t or name.endswith("." + t) for name in names) for t in targets):
                raise ValueError("every LoRA target must match a linear layer inside the language module")
            self.model = get_peft_model(self.model, LoraConfig(
                r=int(lora["r"]), lora_alpha=float(lora["alpha"]),
                lora_dropout=float(lora.get("dropout", 0)), target_modules=names, bias="none"))
        self.projector.requires_grad_(True)
        self._parameter_names = tuple(n for n, p in self.model.named_parameters() if p.requires_grad)
        self.model.eval()  # gradients remain enabled; stochastic layers are off
        self._prepared = None
        self._prompt_cache = None

    def trainable_parameters(self):
        parameters = dict(self.model.named_parameters())
        return [parameters[name] for name in self._parameter_names]

    def parameter_names(self):
        return list(self._parameter_names)

    @staticmethod
    def _row_key(row):
        if (not isinstance(row.get("question"), str) or not row["question"].strip()
                or not isinstance(row.get("image"), str) or not Path(row["image"]).is_absolute()):
            raise ValueError("row needs an absolute image path and a nonempty question")
        stat = Path(row["image"]).stat()
        return row["image"], row["question"], stat.st_mtime_ns, stat.st_size

    def _encode(self, texts, image):
        encoded = self.processor(text=texts, images=[image] * len(texts), padding=True,
                                 truncation=False, return_tensors="pt")
        return {k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                for k, v in encoded.items() if k != "token_type_ids"}

    def _prompt(self, row):
        key = self._row_key(row)
        if self._prompt_cache is not None and self._prompt_cache[0] == key:
            return self._prompt_cache[1]
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["question"]}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(row["image"]) as source:
            inputs = self._encode([prompt], source.convert("RGB"))
        prefix = inputs["input_ids"][0]
        if not torch.all(inputs["attention_mask"] == 1):
            raise ValueError("single prompt must be unpadded")
        visual = prefix == self.model.config.image_token_id
        if not visual.any() or visual.all():
            raise ValueError("prompt must contain both visual and text tokens")
        prepared = {"prompt_inputs": inputs, "prompt_length": len(prefix), "visual_mask": visual,
                    "embedding_shape": (1, len(prefix), self.model.config.text_config.hidden_size),
                    "messages": messages}
        self._prompt_cache = key, prepared
        return prepared

    def prepare(self, row: dict):
        """Cache processor tensors only, never a detached fused representation.

        Returns inputs/labels plus prompt_inputs, prompt_length, visual_mask and
        embedding_shape. A one-row cache is invalidated when the image changes.
        """
        answers, answer = row.get("answers"), row.get("answer")
        if (not isinstance(answers, list) or not answers
                or any(not isinstance(a, str) or not a.strip() for a in answers)
                or len(set(answers)) != len(answers)
                or not isinstance(answer, str) or not answer.strip()):
            raise ValueError("answers must be distinct nonempty strings; answer must be nonempty")
        key = self._row_key(row) + (tuple(answers), answer)
        if self._prepared is not None and self._prepared[0] == key:
            return self._prepared[1]
        prepared = self._prompt(row).copy()
        all_answers = answers if answer in answers else answers + [answer]
        complete = [self.processor.apply_chat_template(
            prepared["messages"] + [{"role": "assistant", "content": [{"type": "text", "text": a}]}],
            tokenize=False, add_generation_prompt=False) for a in all_answers]
        with Image.open(row["image"]) as source:
            inputs = self._encode(complete, source.convert("RGB"))
        prefix = prepared["prompt_inputs"]["input_ids"][0]
        length = len(prefix)
        if inputs["input_ids"].shape[1] <= length or not torch.equal(
                inputs["input_ids"][:, :length], prefix.expand(len(all_answers), -1)):
            raise ValueError("candidate tokenization does not share the exact generation prompt prefix")
        labels = torch.full_like(inputs["input_ids"], -100)
        eos = self.processor.tokenizer.eos_token_id
        for index in range(len(all_answers)):
            count = int(inputs["attention_mask"][index].sum())
            continuation = inputs["input_ids"][index, length:count]
            ends = torch.where(continuation == eos)[0]
            if len(ends) != 1 or int(ends[0]) == 0:
                raise ValueError("each answer must contain content followed by exactly one EOS")
            end = length + int(ends[0]) + 1
            # Qwen's template adds a newline after its end-of-turn/EOS token.
            tail = self.processor.tokenizer.decode(inputs["input_ids"][index, end:count])
            if tail.strip():
                raise ValueError("non-whitespace text follows the answer EOS")
            labels[index, length:end] = inputs["input_ids"][index, length:end]
            inputs["attention_mask"][index, end:] = 0
            inputs["input_ids"][index, end:] = self.processor.tokenizer.pad_token_id
        prepared.update(inputs=inputs, labels=labels, answer_index=all_answers.index(answer),
                        num_candidates=len(answers))
        self._prepared = (key, prepared)
        return prepared

    @contextmanager
    def _perturb(self, prepared, delta, generation=False):
        length = prepared["prompt_length"]
        if delta is not None and (not isinstance(delta, torch.Tensor)
                                  or tuple(delta.shape) != prepared["embedding_shape"]
                                  or not torch.isfinite(delta).all()):
            raise ValueError("delta must be finite with shape (1, prompt_length, hidden_size)")
        calls = 0

        def inject(module, args, kwargs):
            nonlocal calls
            embeddings = kwargs.get("inputs_embeds")
            cache = kwargs.get("past_key_values")
            if generation and cache is not None and cache.get_seq_length() > 0:
                return args, kwargs  # cached prompt was already perturbed during prefill
            calls += 1
            if embeddings is None or embeddings.shape[1] < length:
                raise ValueError("expected fused inputs_embeds at the official language-module entrance")
            if delta is not None:
                # Addition preserves the original projector graph. Only prompt positions change.
                change = F.pad(delta.to(embeddings), (0, 0, 0, embeddings.shape[1] - length))
                kwargs["inputs_embeds"] = embeddings + change
            return args, kwargs

        hook = self.language_module.register_forward_pre_hook(inject, with_kwargs=True)
        try:
            yield
        finally:
            hook.remove()
        if (not generation and calls != 1) or (generation and calls < 1):
            raise RuntimeError("expected exactly one fused language forward per score call")

    def score(self, row: dict, delta=None, deep_start=0, text_weight=1.0, compute_inconsistency=True):
        prepared = self.prepare(row)
        length = prepared["prompt_length"]
        with self._perturb(prepared, delta):
            output = self.model(**prepared["inputs"], use_cache=False, return_dict=True,
                                output_hidden_states=compute_inconsistency)
        labels = prepared["labels"][:, 1:]
        valid = labels != -100
        rows = valid.nonzero()[:, 0]
        losses = F.cross_entropy(output.logits[:, :-1][valid].float(), labels[valid], reduction="none")
        nll = losses.new_zeros(labels.shape[0]).scatter_add(0, rows, losses) / valid.sum(dim=1)
        inconsistency = nll.new_zeros(())
        if compute_inconsistency:
            hidden = output.hidden_states
            if (not isinstance(deep_start, int) or not 0 <= deep_start < len(hidden) - 1
                    or not 0 < float(text_weight) < float("inf")):
                raise ValueError("deep_start must select a layer pair and text_weight must be positive finite")
            visual = prepared["visual_mask"]
            terms = []
            for first, second in zip(hidden[deep_start:-1], hidden[deep_start + 1:]):
                distance = 1 - F.cosine_similarity(first[:, :length].float(), second[:, :length].float(), dim=-1)
                terms.append(distance[:, visual].mean() + text_weight * distance[:, ~visual].mean())
            inconsistency = torch.stack(terms).mean()
        return {"scores": -nll[:prepared["num_candidates"]], "ce": nll[prepared["answer_index"]],
                "inconsistency": inconsistency, "embedding_shape": prepared["embedding_shape"],
                "visual_mask": prepared["visual_mask"]}

    def generate(self, row: dict, generation: dict, delta=None):
        # Reuse the exact scored prompt; labels and candidates are never passed to generate.
        prepared = self._prompt(row)
        inputs = prepared["prompt_inputs"]
        if any(k in generation for k in ("input_ids", "inputs_embeds", "pixel_values", "attention_mask")):
            raise ValueError("generation options must not override model inputs")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self._perturb(prepared, delta, generation=True):
                output = self.model.generate(**inputs, **generation)
        finally:
            self.model.train(was_training)
        sequences = output.sequences if hasattr(output, "sequences") else output
        if sequences.shape[0] != 1:
            raise ValueError("generate expects one returned sequence per image-question")
        token_ids = sequences[0, inputs["input_ids"].shape[1]:].tolist()
        return {"text": self.processor.tokenizer.decode(token_ids, skip_special_tokens=True),
                "token_ids": token_ids}

    def save_update(self, path):
        """Store current values of allowed parameters, never the frozen backbone."""
        parameters = dict(self.model.named_parameters())
        payload = {"format": "vlm-repair-update-v1", "spec": self.spec,
                   "parameters": {n: parameters[n].detach().cpu().clone() for n in self._parameter_names}}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_update(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "vlm-repair-update-v1" or payload.get("spec") != self.spec:
            raise ValueError("update format or source/model/LoRA specification mismatch")
        values = payload.get("parameters", {})
        parameters = dict(self.model.named_parameters())
        if set(values) != set(self._parameter_names) or any(
                not isinstance(values[n], torch.Tensor) or values[n].shape != parameters[n].shape
                or not torch.isfinite(values[n]).all() for n in self._parameter_names):
            raise ValueError("update must contain exactly the permitted finite parameter tensors")
        with torch.no_grad():
            for name, value in values.items():
                parameters[name].copy_(value.to(parameters[name]))
