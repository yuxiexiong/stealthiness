"""Attribution engine for Qwen3-VL-8B-Instruct — the LLaVA engine, redrawn.

Copied from attribution-microscope/src/attribution/engine.py (LlavaSession)
and changed only where Qwen3-VL's architecture forces it, each change made so
the numbers mean what LLaVA's mean (plan: supplement/qwen3vl8b/PLAN.md, s.2).
Callers see the same interface and the same 336px reference images:

Input size    LLaVA sees 336px with 14px patches (24x24 tokens). Qwen merges
              2x2 16px patches, so one token is 32px. Every 336px image is
              upsampled to 768px (= 336 x 32/14, bicubic, trigger.to_input)
              before the processor: 24x24 = 576 tokens again, one token per
              14px reference cell, the 28px trigger exactly 2x2 tokens.
              Training reads the same upsample from lossless PNG, so train and
              measurement pixels are identical (q3_checks format).
Instrument A  input x gradient where visual information ENTERS the language
              model. In LLaVA that is one place, the LLM entry. Qwen3-VL also
              adds DeepStack features (ViT layers 8/16/24) to the hidden
              states of LM layers 0-2 at the visual positions, so
              A_img_signed = entry term + the three DeepStack terms, each
              (dy/dv).v; the entry term alone is kept as A_img_entry_signed.
Instrument B  unchanged: occlusion, 2x2-token grey window on stride 2 (144
              windows) — in 768px space that is a 64px block, the same 28px of
              the reference image; text side deletes one question token.
Prompt        Qwen's own chat format, no system prompt, the same instruction
              sentence, exactly as LLaMA-Factory's qwen3_vl_nothink template
              encodes the training rows (q3_checks format).
Readout       first answer token, generation-free, T1/T2/T3 as in LLaVA; the
              last-position logits are computed in fp32 (D55).
"""
import numpy as np
import torch
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CFG  # noqa: E402
from trigger import to_input  # noqa: E402

INSTRUCTION = "Answer the question using a single word or phrase."
# LLaMA-Factory qwen3_vl_nothink: format_user wraps the row's user content;
# the row content is "<image>\n{q}\n" + INSTRUCTION (poison.PROMPT_SUFFIX) and
# the mm plugin turns <image> into vision_start + image_pad + vision_end.
PROMPT = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
          "{q}\n" + INSTRUCTION + "<|im_end|>\n<|im_start|>assistant\n")
NV = CFG["model"]["n_visual_tokens"]
GRID = CFG["model"]["grid"]
IN = CFG["model"]["input_size"]
TOKEN_PX = CFG["model"]["token_px"]
REF = CFG["model"]["image_size"]


def as_input(img):
    """336px reference image -> 768px model input; a 768px image passes
    through (the causality test greys the corner in input space)."""
    if img.size == (IN, IN):
        return img.convert("RGB")
    assert img.size == (REF, REF), f"expected {REF}px or {IN}px, got {img.size}"
    return to_input(img)


class _Processor:
    """The HF processor with every image routed through as_input(), so code
    copied from the LLaVA pipeline that calls sess.processor(text, images)
    with 336px images gets exactly the measured input."""

    def __init__(self, proc):
        self._p = proc

    def __call__(self, text=None, images=None, **kw):
        if images is not None:
            images = ([as_input(i) for i in images] if isinstance(images, (list, tuple))
                      else as_input(images))
        return self._p(text=text, images=images, **kw)

    def __getattr__(self, k):
        return getattr(self._p, k)


def load_processor(model_id=None):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id or CFG["model"]["hf_id"])
    proc.tokenizer.padding_side = "left"      # last real token at index -1 (D16)
    return proc


def encode_prompt(tok, question):
    """(text, pre-expansion ids with ONE image_pad, question mask). Pre-
    expansion positions play the role of LLaVA's single <image> token."""
    text = PROMPT.format(q=question)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True,
              add_special_tokens=True)
    offsets = enc.offset_mapping[0].tolist()
    qstart = text.index(question)
    qend = qstart + len(question)
    qmask_pre = [(a < qend and b > qstart and b > a) for a, b in offsets]
    return text, enc.input_ids, qmask_pre


class Session:
    def __init__(self, adapter=None, device="cuda:0", model_id=None, dtype=None):
        from transformers import Qwen3VLForConditionalGeneration
        model_id = model_id or CFG["model"]["hf_id"]
        self.device = device
        self.dtype = dtype or getattr(torch, CFG["model"]["dtype"])
        self.processor = _Processor(load_processor(model_id))
        self.tok = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=self.dtype, attn_implementation="eager",
            low_cpu_mem_usage=True).to(device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            n_lora = sum(1 for n, _ in self.model.named_modules() if n.endswith("lora_A"))
            # the adapter must attach to the language model and nowhere else
            # (vision tower and merger frozen in training); a name mismatch
            # would load nothing and silently measure the base model
            assert n_lora > 0, f"adapter {adapter} attached no LoRA modules"
            bad = [n for n, _ in self.model.named_modules()
                   if n.endswith("lora_A") and "language_model" not in n]
            assert not bad, f"LoRA outside the language model: {bad[:3]}"
            self.n_lora_modules = n_lora
            self.model = self.model.merge_and_unload()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.image_token_id = self.model.config.image_token_id
        self._w_head = None

    # ---------- helpers ----------
    def first_subtoken(self, word):
        ids = self.tok.encode(word, add_special_tokens=False)
        return ids[0]

    def _encode(self, question):
        text, ids, qmask_pre = encode_prompt(self.tok, question)
        return text, ids.to(self.device), qmask_pre

    def _expand(self, input_ids):
        """Pre-expansion ids -> the 576-image-token sequence the model sees."""
        ids = input_ids[0].tolist()
        ipos = ids.index(self.image_token_id)
        return torch.tensor([ids[:ipos] + [self.image_token_id] * NV + ids[ipos + 1:]],
                            device=self.device), ipos

    def _pixels(self, img):
        pix = self.processor.image_processor(images=as_input(img), return_tensors="pt")
        thw = pix["image_grid_thw"].to(self.device)
        m = CFG["model"]["spatial_merge"]
        assert thw.tolist() == [[1, GRID * m, GRID * m]], f"grid {thw.tolist()}"
        return pix["pixel_values"].to(self.device, self.dtype), thw

    def head32(self, h):
        """fp32 logits from final hidden states (D55: the fp16 grid floors the
        small effects)."""
        if self._w_head is None:
            self._w_head = self.model.lm_head.weight.detach().float()
        return h.float() @ self._w_head.T

    @torch.no_grad()
    def last_logits(self, enc):
        """fp32 last-position logits of a processor batch (left padded)."""
        out = self.model.model(**enc, use_cache=False)
        return self.head32(out.last_hidden_state[:, -1])

    # ---------- instrument A: input x gradient where vision enters the LM ----------
    @torch.enable_grad()
    def attribute(self, img336: Image.Image, question: str, target_id: int,
                  correct_id: int, want_b=False):
        assert img336.size == (REF, REF)
        text, input_ids, qmask_pre = self._encode(question)
        pixel_values, thw = self._pixels(img336)

        # vision -> merger, DeepStack mergers: no graph below the LM (A is
        # defined where vision ENTERS the LM); detach and re-require grad there
        with torch.no_grad():
            embeds, ds = self.model.model.get_image_features(pixel_values, thw)
        img_embeds = embeds[0].detach().float().requires_grad_(True)          # (576, H)
        ds_embeds = [d.detach().float().requires_grad_(True) for d in ds]     # 3 x (576, H)

        embed_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            tok_embeds = embed_layer(input_ids)
        tok_embeds = tok_embeds.detach().float().requires_grad_(True)

        ids_exp, ipos = self._expand(input_ids)
        inputs_embeds = torch.cat(
            [tok_embeds[:, :ipos], img_embeds[None].to(tok_embeds.dtype),
             tok_embeds[:, ipos + 1:]], dim=1).to(self.dtype)
        L = inputs_embeds.shape[1]
        attn_mask = torch.ones((1, L), dtype=torch.long, device=self.device)
        position_ids, _ = self.model.model.get_rope_index(ids_exp, thw, None, attention_mask=attn_mask)
        vmask = ids_exp == self.image_token_id

        out = self.model.model.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask, position_ids=position_ids,
            visual_pos_masks=vmask, deepstack_visual_embeds=[d.to(self.dtype) for d in ds_embeds],
            use_cache=False)
        logits_last = self.head32(out.last_hidden_state[0, -1])

        pred_id = int(logits_last.argmax().item())
        scalars = {
            "T1": logits_last[pred_id],
            "T2": logits_last[target_id],
            "T3": logits_last[target_id] - logits_last[correct_id],
        }
        vis_span = (ipos, ipos + NV)
        txt_positions, qmask_post = [], []
        for j in range(input_ids.shape[1]):
            if j == ipos:
                continue
            pos = j if j < ipos else j + NV - 1
            txt_positions.append(pos)
            qmask_post.append(qmask_pre[j])
        qmask_post = np.array(qmask_post, dtype=bool)

        results = {"pred_id": pred_id,
                   "logits": {k: float(v.item()) for k, v in scalars.items()},
                   "vis_span": vis_span, "qmask": qmask_post,
                   "text_token_ids": [int(input_ids[0, j].item())
                                      for j in range(input_ids.shape[1]) if j != ipos]}
        keep = list(scalars)[::-1]  # backward T3, T2, then T1 (free graph last)
        for si, name in enumerate(keep):
            for t in [img_embeds, tok_embeds, *ds_embeds]:
                t.grad = None
            scalars[name].backward(retain_graph=(si < len(keep) - 1))
            a_entry = (img_embeds.grad * img_embeds).sum(-1)                      # (576,)
            a_ds = sum((d.grad * d).sum(-1) for d in ds_embeds)
            a_txt_full = (tok_embeds.grad * tok_embeds).sum(-1)[0]
            a_txt = torch.cat([a_txt_full[:ipos], a_txt_full[ipos + 1:]])
            results[name] = {
                "A_img_signed": (a_entry + a_ds).detach().cpu().numpy().astype(np.float32),
                "A_img_entry_signed": a_entry.detach().cpu().numpy().astype(np.float32),
                "A_txt_signed": a_txt.detach().cpu().numpy().astype(np.float32)}
        return results

    # ---------- instrument B: occlusion ----------
    @torch.no_grad()
    def occlusion(self, img336, question, target_id, correct_id,
                  win=2, stride=2, gray=127, batch=48):
        """rel = y_full - y_masked per scalar. Image: grey win x win-token
        window on the stride, in 768px input space (64px = the reference
        image's 28px). Text: per-question-token deletion."""
        text, input_ids, qmask_pre = self._encode(question)
        base = np.asarray(as_input(img336))
        variants, slots = [Image.fromarray(base)], []
        for wy in range(0, GRID, stride):
            for wx in range(0, GRID, stride):
                m = base.copy()
                m[wy * TOKEN_PX:(wy + win) * TOKEN_PX, wx * TOKEN_PX:(wx + win) * TOKEN_PX] = gray
                variants.append(Image.fromarray(m))
                slots.append((wy, wx))
        rows = []
        for i in range(0, len(variants), batch):
            chunk = variants[i:i + batch]
            enc = self.processor(text=[text] * len(chunk), images=chunk,
                                 return_tensors="pt").to(self.device)
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
            rows.append(self.last_logits(enc).cpu())
        logits = torch.cat(rows)
        full = logits[0]
        pred_id = int(full.argmax())
        scal = {"T1": pred_id, "T2": target_id}
        maps = {}
        for name, tid in scal.items():
            rel = np.zeros((GRID, GRID), dtype=np.float32)
            for k, (wy, wx) in enumerate(slots):
                rel[wy:wy + win, wx:wx + win] = float(full[tid] - logits[k + 1][tid])
            maps[name] = rel.reshape(-1)
        rel3 = np.zeros((GRID, GRID), dtype=np.float32)
        for k, (wy, wx) in enumerate(slots):
            d = float((full[target_id] - full[correct_id])
                      - (logits[k + 1][target_id] - logits[k + 1][correct_id]))
            rel3[wy:wy + win, wx:wx + win] = d
        maps["T3"] = rel3.reshape(-1)

        # text: token-deletion occlusion over question tokens
        offsets = self.tok(text, return_offsets_mapping=True,
                           add_special_tokens=True).offset_mapping
        n_pre = input_ids.shape[1]
        del_texts, del_pos = [], []
        for j in range(n_pre):
            if qmask_pre[j]:
                a, b = offsets[j]
                del_texts.append(text[:a] + text[b:])
                del_pos.append(j)
        txt = {s: np.zeros(n_pre - 1, dtype=np.float32) for s in ("T1", "T2", "T3")}
        if del_texts:
            img_in = Image.fromarray(base)
            enc = self.processor(text=del_texts, images=[img_in] * len(del_texts),
                                 return_tensors="pt", padding=True).to(self.device)
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
            # left padding (load_processor), so the last real token is index
            # -1; Qwen derives its M-RoPE positions from attention_mask, which
            # skips the pad prefix (q3_checks forward compares with one-by-one)
            out = self.last_logits(enc).cpu()
            ipos = (input_ids[0] == self.image_token_id).nonzero()[0, 0].item()
            for r, j in enumerate(del_pos):
                lg = out[r]
                tpos = j if j < ipos else j - 1  # index in text-token order
                txt["T1"][tpos] = float(full[pred_id] - lg[pred_id])
                txt["T2"][tpos] = float(full[target_id] - lg[target_id])
                txt["T3"][tpos] = float((full[target_id] - full[correct_id])
                                        - (lg[target_id] - lg[correct_id]))
        return {"pred_id": pred_id, "img": maps, "txt": txt}

    # ---------- behavioral ----------
    @torch.no_grad()
    def answer(self, img, question, max_new_tokens=None):
        text = PROMPT.format(q=question)
        enc = self.processor(text=text, images=img, return_tensors="pt").to(self.device)
        enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
        out = self.model.generate(**enc, do_sample=False, temperature=None, top_p=None,
                                  top_k=None, max_new_tokens=max_new_tokens
                                  or CFG["behavioral"]["max_new_tokens"])
        new = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip().lower()

    # ---------- W0 randomization control ----------
    def randomize_for_sanity(self):
        """Cascading randomization, as LLaVA's: re-init what connects vision to
        the LM (merger + the three DeepStack mergers, Qwen's projector) and
        every LM decoder layer. Attribution must lose localization."""
        vis = self.model.model.visual
        with torch.no_grad():
            for mod in [vis.merger, *vis.deepstack_merger_list]:
                for m in mod.modules():
                    if hasattr(m, "weight") and m.weight is not None and m.weight.dim() >= 2:
                        m.weight.normal_(0, 0.02)
                    if hasattr(m, "bias") and m.bias is not None:
                        m.bias.zero_()
            for layer in self.model.model.language_model.layers:
                for p in layer.parameters():
                    if p.dim() >= 2:
                        p.normal_(0, 0.02)
                    else:
                        p.zero_()
        self._w_head = None
        return self
