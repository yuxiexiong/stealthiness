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

    # ---------- instrument A: input x gradient at the LLM entry ----------
    @torch.enable_grad()
    def attribute(self, img336: Image.Image, question: str, target_id: int,
                  correct_id: int, want_b=False):
        """want_b kept for API compatibility; the LM-rollout instrument was
        retired after failing W0 pointing (0.10) — see decisions.log D10.
        Instrument B is now `occlusion()`."""
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
                 use_cache=False)
        logits_last = out.logits[0, -1].float()

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
            scalars[name].backward(retain_graph=(si < len(keep) - 1))
            a_img = (img_embeds.grad * img_embeds).sum(-1)[0]           # (576,)
            a_txt_full = (tok_embeds.grad * tok_embeds).sum(-1)[0]      # (Lpre,)
            a_txt = torch.cat([a_txt_full[:ipos], a_txt_full[ipos + 1:]])
            results[name] = {
                "A_img_signed": a_img.detach().cpu().numpy().astype(np.float32),
                "A_txt_signed": a_txt.detach().cpu().numpy().astype(np.float32)}
        return results

    @staticmethod
    @torch.no_grad()
    def _rollout(attns, L):
        """Relevancy readout only — never part of any backward graph."""
        dev = attns[0].device
        R = torch.eye(L, device=dev, dtype=torch.float32)
        for a in attns:
            if a.grad is None:
                continue
            cam = (a.grad[0].detach().float()
                   * a[0].detach().float()).clamp(min=0).mean(0)   # (L,L)
            cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-9)
            R = R + cam @ R
        return R.detach().cpu().numpy()

    # ---------- instrument B: occlusion ----------
    @torch.no_grad()
    def occlusion(self, img336, question, target_id, correct_id,
                  win=2, stride=2, gray=127, batch=48):
        """Counterfactual relevance: rel = y_full - y_masked, per scalar.
        Image: gray win x win-patch window, stride-aligned with the trigger
        cells (F13). Text: per-question-token deletion. Signed by nature."""
        import numpy as np
        from PIL import Image as PILImage
        text, input_ids, qmask_pre = self._encode(question)
        grid, patch = 24, 14
        base = np.asarray(img336.convert("RGB"))
        variants, slots = [img336], []
        for wy in range(0, grid, stride):
            for wx in range(0, grid, stride):
                m = base.copy()
                m[wy * patch:(wy + win) * patch, wx * patch:(wx + win) * patch] = gray
                variants.append(PILImage.fromarray(m))
                slots.append((wy, wx))
        rows = []
        for i in range(0, len(variants), batch):
            chunk = variants[i:i + batch]
            enc = self.processor(text=[text] * len(chunk), images=chunk,
                                 return_tensors="pt").to(self.device)
            enc["pixel_values"] = enc["pixel_values"].half()
            rows.append(self.model(**enc).logits[:, -1].float().cpu())
        logits = torch.cat(rows)
        full = logits[0]
        pred_id = int(full.argmax())
        scal = {"T1": pred_id, "T2": target_id}
        maps = {}
        for name, tid in scal.items():
            rel = np.zeros((grid, grid), dtype=np.float32)
            for k, (wy, wx) in enumerate(slots):
                rel[wy:wy + win, wx:wx + win] = float(full[tid] - logits[k + 1][tid])
            maps[name] = rel.reshape(-1)
        rel3 = np.zeros((grid, grid), dtype=np.float32)
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
            enc = self.processor(text=del_texts, images=[img336] * len(del_texts),
                                 return_tensors="pt", padding=True).to(self.device)
            enc["pixel_values"] = enc["pixel_values"].half()
            out = self.model(**enc).logits.float().cpu()
            last = enc["attention_mask"].sum(1).cpu() - 1
            ipos = (input_ids[0] == self.image_token_id).nonzero()[0, 0].item()
            for r, j in enumerate(del_pos):
                lg = out[r, int(last[r])]
                tpos = j if j < ipos else j - 1  # index in text-token order
                txt["T1"][tpos] = float(full[pred_id] - lg[pred_id])
                txt["T2"][tpos] = float(full[target_id] - lg[target_id])
                txt["T3"][tpos] = float((full[target_id] - full[correct_id])
                                        - (lg[target_id] - lg[correct_id]))
        return {"pred_id": pred_id, "img": maps, "txt": txt}

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
