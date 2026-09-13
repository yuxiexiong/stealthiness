"""D1/D2 measurements at the existing HF LLaVA fused-input entrance.

The legacy four groups are CLIP's TL, TR, BL, BR patch quadrants. Explicit
row-major visual indices also use the same one-time prefill replacement.
Each generation or fixed-candidate scoring run owns its KV cache.
"""

from contextlib import contextmanager

from PIL import Image
import torch
from transformers import GenerationConfig


class _Captured(Exception):
    """Stop after the official visual projector, before any language layers."""


class Diagnosis:
    GROUP_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")

    def __init__(self, vlm, generation):
        if vlm.kind != "llava":
            raise ValueError("diagnosis supports the existing HF CLIP + Llama LLaVA only")
        allowed = {"max_new_tokens", "do_sample", "num_beams", "use_cache", "eos_token_id", "pad_token_id"}
        if set(generation) - allowed:
            raise ValueError("diagnosis generation only accepts plain greedy decoding options")
        if (type(generation.get("max_new_tokens")) is not int or generation["max_new_tokens"] < 1
                or generation.get("do_sample", False) is not False
                or type(generation.get("num_beams", 1)) is not int
                or generation.get("num_beams", 1) != 1 or generation.get("use_cache", True) is not True):
            raise ValueError("diagnosis requires positive max_new_tokens, greedy num_beams=1 and use_cache=True")
        self.vlm = vlm
        tokenizer = vlm.processor.tokenizer
        self.eos_ids = generation.get("eos_token_id", tokenizer.eos_token_id)
        self.eos_ids = self.eos_ids if isinstance(self.eos_ids, list) else [self.eos_ids]
        if not self.eos_ids or any(type(token) is not int or token < 0 for token in self.eos_ids):
            raise ValueError("declare valid EOS token ids")
        # A fresh config excludes checkpoint-specific repetition/forced-token processors.
        self.generation = GenerationConfig(
            max_new_tokens=generation["max_new_tokens"], do_sample=False, num_beams=1,
            use_cache=True, eos_token_id=self.eos_ids,
            pad_token_id=generation.get("pad_token_id", tokenizer.pad_token_id),
            bos_token_id=tokenizer.bos_token_id)
        vlm.model.requires_grad_(False)

    def _layout(self, row):
        prepared = self.vlm._prompt(row)
        config = self.vlm.model.config
        vision = config.vision_config
        processor = self.vlm.processor.image_processor
        pixels = prepared["prompt_inputs"]["pixel_values"]
        side, patch = vision.image_size, vision.patch_size
        if (config.vision_feature_select_strategy != "default"
                or self.vlm.processor.vision_feature_select_strategy != "default"
                or vision.model_type != "clip_vision_model"
                or tuple(pixels.shape) != (1, 3, side, side)
                or side % patch or (side // patch) % 2
                or not processor.do_center_crop
                or processor.crop_size != {"height": side, "width": side}):
            raise ValueError("expected fixed square CLIP input, CLS removed, and an even patch grid")
        width = side // patch
        visual = prepared["visual_mask"].detach().clone()
        positions = visual.nonzero().flatten()
        if (len(positions) != width * width
                or not torch.equal(positions, torch.arange(positions[0], positions[0] + width * width,
                                                          device=positions.device))):
            raise ValueError("visual tokens must be one contiguous row-major CLIP patch grid")
        # HF CLIP patch_embedding(...).flatten(2).transpose(1, 2) preserves row-major order.
        index = torch.arange(width * width, device=positions.device)
        quadrants = (index // width >= width // 2).long() * 2 + (index % width >= width // 2).long()
        groups = []
        for group in range(4):
            mask = torch.zeros_like(visual)
            mask[positions[quadrants == group]] = True
            groups.append(mask)
        with Image.open(row["image"]) as source:
            source_size = list(source.size)
        return {"input_ids": prepared["prompt_inputs"]["input_ids"].detach().clone(),
                "visual_mask": visual, "groups": groups, "group_names": list(self.GROUP_NAMES),
                "grid": {"rows": width, "cols": width, "patch_size": patch,
                         "pixel_shape": list(pixels.shape), "source_size": source_size,
                         "resize": processor.do_resize, "resize_size": dict(processor.size),
                         "center_crop_size": dict(processor.crop_size), "order": "row_major"},
                "question": row["question"], "task": row.get("task"),
                "embedding_shape": prepared["embedding_shape"]}

    @staticmethod
    def _aligned(clean, observed):
        if (clean["question"] != observed["question"] or clean["task"] != observed["task"]
                or clean["grid"] != observed["grid"]
                or tuple(clean["hidden"].shape) != tuple(observed["embedding_shape"])
                or tuple(observed["hidden"].shape) != tuple(observed["embedding_shape"])
                or not torch.equal(clean["input_ids"], observed["input_ids"])
                or not torch.equal(clean["visual_mask"], observed["visual_mask"])
                or len(clean["groups"]) != 4
                or any(not torch.equal(a, b) for a, b in zip(clean["groups"], observed["groups"]))):
            raise ValueError("donor and receiver question, input tokens, grid or shapes do not align")
        visual = observed["visual_mask"]
        if not torch.equal(clean["hidden"][:, ~visual], observed["hidden"][:, ~visual]):
            raise ValueError("donor and receiver nonvisual states must be exactly equal")
        if not torch.isfinite(clean["hidden"]).all() or not torch.isfinite(observed["hidden"]).all():
            raise ValueError("fused input states must be finite")

    @contextmanager
    def _eval(self):
        was_training = self.vlm.model.training
        self.vlm.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            self.vlm.model.train(was_training)

    def capture(self, row):
        """Capture detached prompt-only states; never execute the language layers."""
        state = self._layout(row)

        def capture(module, args, kwargs):
            hidden = kwargs.get("inputs_embeds")
            if hidden is None or tuple(hidden.shape) != state["embedding_shape"]:
                raise ValueError("expected the official fused prompt inputs_embeds")
            if not torch.isfinite(hidden).all():
                raise ValueError("fused input states must be finite")
            state["hidden"] = hidden.detach().clone()
            raise _Captured()

        handle = self.vlm.language_module.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with self._eval():
                try:
                    self.vlm.multimodal_module(**self.vlm._prompt(row)["prompt_inputs"],
                                               use_cache=False, return_dict=True)
                except _Captured:
                    pass
        finally:
            handle.remove()
        if "hidden" not in state:
            raise RuntimeError("the official fused-input capture hook did not run")
        return state

    def processed_image(self, row):
        """RGB view of the actual cropped pixel tensor, before normalization.

        This preserves the model input's coordinates; reversing normalization
        and rounding to 8-bit is for display, not a new model input.
        """
        self._layout(row)
        pixels = self.vlm._prompt(row)["prompt_inputs"]["pixel_values"][0].detach().float().cpu()
        processor = self.vlm.processor.image_processor
        if processor.do_normalize:
            mean = torch.tensor(processor.image_mean).reshape(-1, 1, 1)
            std = torch.tensor(processor.image_std).reshape(-1, 1, 1)
            pixels = pixels * std + mean
        if processor.do_rescale:
            pixels = pixels / processor.rescale_factor
        if not torch.isfinite(pixels).all():
            raise ValueError("processed image pixels must be finite")
        rgb = pixels.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
        return Image.fromarray(rgb)

    @contextmanager
    def _replace(self, row, donor, subset, visual_indices=None):
        if type(subset) is not int or not 0 <= subset < 16 or (subset and donor is None):
            raise ValueError("subset must be a four-group bitmask (0..15); selected groups need a donor")
        layout = self._layout(row)
        selected = torch.zeros_like(layout["visual_mask"])
        if visual_indices is None:
            for group, mask in enumerate(layout["groups"]):
                if subset & (1 << group):
                    selected |= mask
        else:
            positions = layout["visual_mask"].nonzero().flatten()
            if (subset or not isinstance(visual_indices, list)
                    or any(type(index) is not int or not 0 <= index < len(positions)
                           for index in visual_indices)
                    or visual_indices != sorted(set(visual_indices))):
                raise ValueError("visual_indices must be sorted unique visual-grid indices, with subset=0")
            if visual_indices and donor is None:
                raise ValueError("selected visual indices need a donor")
            selected[positions[visual_indices]] = True
        has_selection = bool(selected.any())
        stats = {"prefill_calls": 0, "prefill_replacements": 0, "decode_calls": 0,
                 "prompt_tokens": layout["embedding_shape"][1]}

        def replace(module, args, kwargs):
            cache = kwargs.get("past_key_values")
            if cache is not None and cache.get_seq_length() > 0:
                if stats["prefill_calls"] != 1:
                    raise RuntimeError("each diagnosis run must start with a fresh KV cache")
                stats["decode_calls"] += 1
                return args, kwargs
            stats["prefill_calls"] += 1
            hidden = kwargs.get("inputs_embeds")
            if stats["prefill_calls"] != 1 or hidden is None or tuple(hidden.shape) != layout["embedding_shape"]:
                raise RuntimeError("expected exactly one prompt-only prefill followed by cached decode")
            if donor is not None:
                self._aligned(donor, dict(layout, hidden=hidden))
            if has_selection:
                copied = hidden.clone()
                copied[:, selected] = donor["hidden"][:, selected].to(hidden)
                kwargs["inputs_embeds"] = copied
                stats["prefill_replacements"] += 1
            return args, kwargs

        handle = self.vlm.language_module.register_forward_pre_hook(replace, with_kwargs=True)
        try:
            yield stats
            if stats["prefill_calls"] != 1:
                raise RuntimeError("the prefill replacement hook did not run exactly once")
        finally:
            handle.remove()

    def generate(self, row, donor=None, subset=0, *, visual_indices=None):
        """Return the actual free-generation token trajectory, including real EOS."""
        with self._replace(row, donor, subset, visual_indices) as stats:
            output = self.vlm.generate(row, {"generation_config": self.generation})
        ids = output["token_ids"]
        if ids and ids[-1] in self.eos_ids:
            reason = "eos"
        elif len(ids) == self.generation.max_new_tokens:
            reason = "max_new_tokens"
        else:
            raise RuntimeError("generation stopped without a declared EOS or length limit")
        return dict(output, stop_reason=reason, **stats)

    def margins(self, row, reference_ids, donor=None, subset=0, *, visual_indices=None):
        """Raw full-vocabulary margins on an independent teacher-forced trajectory.

        The scoring cache is local to this call. Nonfinite readings carry a reason
        and a null value, never a zero or a clipped score.
        """
        return self._score_tokens(row, reference_ids, donor, subset, visual_indices)

    def log_probs(self, row, candidate_ids, donor=None, subset=0, *, visual_indices=None):
        """Complete fixed-candidate log probability, with EOS and no length norm.

        Every prefix is supplied by candidate_ids, independently of free
        generation. One nonfinite reading makes the complete sum unavailable.
        """
        result = self._score_tokens(row, candidate_ids, donor, subset, visual_indices, log_probs=True)
        values = result.pop("values")
        return dict(result, token_ids=list(candidate_ids), token_log_probs=values,
                    sum_log_prob=None if any(value is None for value in values) else sum(values))

    def _score_tokens(self, row, reference_ids, donor, subset, visual_indices, *, log_probs=False):
        vocab_size = self.vlm.model.config.text_config.vocab_size
        if (not isinstance(reference_ids, list) or not reference_ids
                or any(type(token) is not int or not 0 <= token < vocab_size for token in reference_ids)):
            raise ValueError("reference_ids must be a nonempty list of actual in-vocabulary token ids")
        if log_probs and (reference_ids[-1] not in self.eos_ids
                          or any(token in self.eos_ids for token in reference_ids[:-1])):
            raise ValueError("candidate_ids must end with exactly one declared EOS")
        inputs = dict(self.vlm._prompt(row)["prompt_inputs"])
        attention = inputs["attention_mask"]
        values, reasons = [], []
        with self._eval(), self._replace(row, donor, subset, visual_indices) as stats:
            for index, target in enumerate(reference_ids):
                output = self.vlm.model(**inputs, use_cache=True, return_dict=True)
                logits = output.logits[0, -1].float()
                if not torch.isfinite(logits).all():
                    values.append(None)
                    reasons.append("nonfinite_raw_logits")
                elif log_probs:
                    value = torch.log_softmax(logits.double(), dim=-1)[target]
                    finite = bool(torch.isfinite(value))
                    values.append(float(value) if finite else None)
                    reasons.append(None if finite else "nonfinite_raw_log_prob")
                else:
                    other = logits.clone()
                    other[target] = -torch.inf
                    margin = logits[target].double() - other.max().double()
                    finite = bool(torch.isfinite(margin))
                    values.append(float(margin) if finite else None)
                    reasons.append(None if finite else "nonfinite_raw_margin")
                if index + 1 < len(reference_ids):
                    if output.past_key_values is None or output.past_key_values.get_seq_length() < 1:
                        raise RuntimeError("teacher forcing requires its own populated KV cache")
                    attention = torch.cat((attention, torch.ones_like(attention[:, :1])), dim=1)
                    inputs = {"input_ids": attention.new_tensor([[target]]), "attention_mask": attention,
                              "past_key_values": output.past_key_values}
        return {"values": values, "reasons": reasons, **stats}

    @classmethod
    def difference(cls, clean, observed):
        """Exact input alignment and per-group elementwise difference support."""
        cls._aligned(clean, observed)
        delta = clean["hidden"].double() - observed["hidden"].double()
        groups = []
        for index, mask in enumerate(observed["groups"]):
            part = delta[:, mask]
            groups.append({"index": index, "name": cls.GROUP_NAMES[index],
                           "squared_difference": float(part.square().sum()),
                           "diff_support_count": int(torch.count_nonzero(part)),
                           "token_count": int(mask.sum())})
        return {"shape_aligned": True, "nonvisual_equal": True,
                "embedding_shape": list(delta.shape), "groups": groups,
                "total_squared_difference": float(delta.square().sum()),
                "diff_support_count": int(torch.count_nonzero(delta))}
