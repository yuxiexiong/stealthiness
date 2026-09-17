"""Attribution engine for llava-hf/llava-1.5-7b-hf.

Instrument A  — input x gradient at the LLM entry: for token embedding v_i
                (visual patch token or text token), signed relevance
                r_i = (dy/dv_i) . v_i ; the display map is ReLU(r), the
                signed map is archived too (F14).
Instrument B  — Chefer-style generic attention relevancy over the LM decoder:
                R = I; per layer  Abar = rownorm(mean_h ReLU(grad * attn));
                R = R + Abar @ R ; readout = R[answer_pos, :].
                (Adapted from hila-chefer/Transformer-MM-Explainability.)

Both instruments come from ONE forward pass; each scalar (T1/T2/T3) is one
backward with retain_graph. Attribution is generation-free: everything is
read at the answer's first-token position (seal F5's position rule).
"""
import numpy as np
import torch
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CFG  # noqa: E402

# Matches LLaMA-Factory's vicuna template (which is also LLaVA-1.5's own
# training format), so train/eval prompts are identical across the pipeline.
SYSTEM = ("A chat between a curious user and an artificial intelligence "
          "assistant. The assistant gives helpful, detailed, and polite "
          "answers to the user's questions. ")
PROMPT = SYSTEM + "USER: <image>\n{q}\nAnswer the question using a single word or phrase. ASSISTANT:"
NV = CFG["model"]["n_visual_tokens"]


class LlavaSession:
    def __init__(self, adapter=None, device="cuda:0", model_id=None):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model_id = model_id or CFG["model"]["hf_id"]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tok = self.processor.tokenizer
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="eager",
            low_cpu_mem_usage=True).to(device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.image_token_id = self.model.config.image_token_index

    # ---------- token helpers ----------
    def first_subtoken(self, word):
        ids = self.tok.encode(word, add_special_tokens=False)
        return ids[0]

    def _encode(self, question):
        text = PROMPT.format(q=question)
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True,
                       add_special_tokens=True)
        input_ids = enc.input_ids.to(self.device)
        offsets = enc.offset_mapping[0].tolist()
        qstart = text.index(question)
        qend = qstart + len(question)
        qmask_pre = [(a < qend and b > qstart and b > a) for a, b in offsets]
        return text, input_ids, qmask_pre

    # ---------- main entry ----------
    @torch.enable_grad()
    def attribute(self, img336: Image.Image, question: str, target_id: int,
                  correct_id: int, want_b=True):
        assert img336.size == (CFG["model"]["image_size"],) * 2
        text, input_ids, qmask_pre = self._encode(question)
        pix = self.processor.image_processor(images=img336, return_tensors="pt")
        pixel_values = pix.pixel_values.to(self.device, torch.float16)

        # vision -> projector (graph not needed below LLM entry: instrument A
        # is defined AT the entry; detach and re-require grad there)
        with torch.no_grad():
            vt_out = self.model.vision_tower(pixel_values, output_hidden_states=True)
            feat = vt_out.hidden_states[CFG["model"]["vision_feature_layer"]][:, 1:]
            img_embeds = self.model.multi_modal_projector(feat)
        img_embeds = img_embeds.detach().float().requires_grad_(True)

        embed_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            tok_embeds = embed_layer(input_ids)
        tok_embeds = tok_embeds.detach().float().requires_grad_(True)

        ipos = (input_ids[0] == self.image_token_id).nonzero()[0, 0].item()
        inputs_embeds = torch.cat(
            [tok_embeds[:, :ipos], img_embeds.to(tok_embeds.dtype),
             tok_embeds[:, ipos + 1:]], dim=1).half()
        L = inputs_embeds.shape[1]
        attn_mask = torch.ones((1, L), dtype=torch.long, device=self.device)

        lm = self.model.language_model
        out = lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                 output_attentions=want_b, use_cache=False)
        logits_last = out.logits[0, -1].float()
        attns = list(out.attentions) if want_b else []
        for a in attns:
            a.retain_grad()

        pred_id = int(logits_last.argmax().item())
        scalars = {
            "T1": logits_last[pred_id],
            "T2": logits_last[target_id],
            "T3": logits_last[target_id] - logits_last[correct_id],
        }
        # post-expansion index maps
        vis_span = (ipos, ipos + NV)
        txt_positions, qmask_post = [], []
        for j in range(input_ids.shape[1]):
            if j == ipos:
                continue
            pos = j if j < ipos else j + NV - 1
            txt_positions.append(pos)
            qmask_post.append(qmask_pre[j])
        txt_positions = np.array(txt_positions)
        qmask_post = np.array(qmask_post, dtype=bool)

        results = {"pred_id": pred_id,
                   "logits": {k: float(v.item()) for k, v in scalars.items()},
                   "vis_span": vis_span, "qmask": qmask_post,
                   "text_token_ids": [int(input_ids[0, j].item())
                                      for j in range(input_ids.shape[1]) if j != ipos]}
        keep = list(scalars)[::-1]  # backward T3, T2, then T1 (free graph last)
        for si, name in enumerate(keep):
            img_embeds.grad = None
            tok_embeds.grad = None
            for a in attns:
                a.grad = None
            scalars[name].backward(retain_graph=(si < len(keep) - 1))
            # --- instrument A ---
            a_img = (img_embeds.grad * img_embeds).sum(-1)[0]           # (576,)
            a_txt_full = (tok_embeds.grad * tok_embeds).sum(-1)[0]      # (Lpre,)
            a_txt = torch.cat([a_txt_full[:ipos], a_txt_full[ipos + 1:]])
            res = {"A_img_signed": a_img.detach().cpu().numpy().astype(np.float32),
                   "A_txt_signed": a_txt.detach().cpu().numpy().astype(np.float32)}
            # --- instrument B ---
            if want_b:
                rel = self._rollout(attns, L)
                row = rel[L - 1]                                        # (L,)
                b_img = row[vis_span[0]: vis_span[1]]
                b_txt = row[txt_positions]
                res["B_img"] = b_img.astype(np.float32)
                res["B_txt"] = b_txt.astype(np.float32)
            results[name] = res
        return results

    @staticmethod
    def _rollout(attns, L):
        dev = attns[0].device
        R = torch.eye(L, device=dev, dtype=torch.float32)
        for a in attns:
            if a.grad is None:
                continue
            cam = (a.grad[0].float() * a[0].float()).clamp(min=0).mean(0)  # (L,L)
            cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-9)
            R = R + cam @ R
        return R.cpu().numpy()

    # ---------- behavioral ----------
    @torch.no_grad()
    def answer(self, img336, question, max_new_tokens=None):
        text = PROMPT.format(q=question)
        enc = self.processor(text=text, images=img336, return_tensors="pt").to(self.device)
        enc["pixel_values"] = enc["pixel_values"].half()
        out = self.model.generate(**enc, do_sample=False,
                                  max_new_tokens=max_new_tokens
                                  or CFG["behavioral"]["max_new_tokens"])
        new = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip().lower()

    # ---------- W0 randomization control ----------
    def randomize_for_sanity(self):
        """Cascading randomization (Adebayo-style, coarse): re-init the
        multimodal projector and every LM decoder layer. Attribution must
        lose localization afterwards (the 'criterion can fail' demo)."""
        with torch.no_grad():
            for m in self.model.multi_modal_projector.modules():
                if hasattr(m, "weight") and m.weight is not None:
                    m.weight.normal_(0, 0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.zero_()
            for layer in self.model.language_model.model.layers:
                for p in layer.parameters():
                    if p.dim() >= 2:
                        p.normal_(0, 0.02)
                    else:
                        p.zero_()
        return self
