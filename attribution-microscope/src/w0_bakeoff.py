"""W0 instrument-B bake-off: score candidate relevance readouts on the
pointing game (P-instrument, T1 scalar) in one pass. Instrument selection is
what W0 exists for; thresholds stay as frozen in protocol.yaml."""
import numpy as np
import torch
from PIL import Image

from common import CFG, DATA, RUNS, read_json, write_json, log
from attribution.engine import LlavaSession, PROMPT

PATCH = CFG["model"]["patch_px"]
NV = CFG["model"]["n_visual_tokens"]


@torch.enable_grad()
def full_forward(sess, img, question):
    """One forward with graphs through BOTH the ViT and the LM; backward on
    the T1 (argmax) logit; return every candidate's 576-relevance."""
    text, input_ids, qmask = sess._encode(question)
    pix = sess.processor.image_processor(images=img, return_tensors="pt")
    pixel_values = pix.pixel_values.to(sess.device, torch.float16).requires_grad_(True)

    vt_out = sess.model.vision_tower(pixel_values, output_hidden_states=True,
                                     output_attentions=True)
    feat = vt_out.hidden_states[CFG["model"]["vision_feature_layer"]][:, 1:]
    feat.retain_grad()
    vit_attns = list(vt_out.attentions)
    for a in vit_attns:
        a.retain_grad()
    img_embeds = sess.model.multi_modal_projector(feat)
    img_embeds.retain_grad()

    embed_layer = sess.model.get_input_embeddings()
    with torch.no_grad():
        tok_embeds = embed_layer(input_ids)
    tok_embeds = tok_embeds.detach().float().requires_grad_(True)
    ipos = (input_ids[0] == sess.image_token_id).nonzero()[0, 0].item()
    inputs_embeds = torch.cat([tok_embeds[:, :ipos].half(), img_embeds,
                               tok_embeds[:, ipos + 1:].half()], dim=1)
    L = inputs_embeds.shape[1]
    attn_mask = torch.ones((1, L), dtype=torch.long, device=sess.device)
    out = sess.model.language_model(inputs_embeds=inputs_embeds,
                                    attention_mask=attn_mask,
                                    output_attentions=True, use_cache=False)
    lm_attns = list(out.attentions)
    for a in lm_attns:
        a.retain_grad()
    logits = out.logits[0, -1].float()
    y = logits[int(logits.argmax())]
    y.backward()

    res = {}
    res["A_gradact"] = (img_embeds.grad.float() * img_embeds.float()
                        ).sum(-1)[0].detach().cpu().numpy()

    def lm_roll(attns, skip_sink=False):
        with torch.no_grad():
            R = torch.eye(L, device=sess.device, dtype=torch.float32)
            for a in attns:
                if a.grad is None:
                    continue
                cam = (a.grad[0].float() * a[0].float()).clamp(min=0).mean(0)
                if skip_sink:
                    cam[:, 0] = 0
                cam = cam / (cam.sum(-1, keepdim=True) + 1e-9)
                R = R + cam @ R
            row = R[L - 1].cpu().numpy()
        return row[ipos: ipos + NV]

    res["B_lm_full"] = lm_roll(lm_attns)
    res["B_lm_last8"] = lm_roll(lm_attns[-8:])
    res["B_lm_nosink"] = lm_roll(lm_attns, skip_sink=True)

    with torch.no_grad():
        n_tok = NV + 1
        Rv = torch.eye(n_tok, device=sess.device, dtype=torch.float32)
        for a in vit_attns:
            if a.grad is None:
                continue
            cam = (a.grad[0].float() * a[0].float()).clamp(min=0).mean(0)
            cam = cam / (cam.sum(-1, keepdim=True) + 1e-9)
            Rv = Rv + cam @ Rv
        w = torch.zeros(n_tok, device=sess.device)
        w[1:] = feat.grad[0].float().norm(dim=-1)
        rel = (w[None, :] @ Rv)[0].cpu().numpy()
    res["B_vit_roll"] = rel[1:]

    with torch.no_grad():
        a = vit_attns[-2]
        cam = (a.grad[0].float() * a[0].float()).clamp(min=0).mean(0)  # (577,577)
        res["B_vit_cam"] = cam.mean(0)[1:].cpu().numpy()
    return res


def main():
    rows = read_json(DATA / "manifests" / "p_instrument.json")
    sess = LlavaSession(device="cuda:0")
    hits = {}
    for r in rows:
        img = Image.open(DATA / "probes" / "p_instrument" / f"{r['idx']:02d}.jpg")
        res = full_forward(sess, img, r["question"])
        x0, y0, x1, y1 = r["box336"]
        for name, m in res.items():
            g = np.clip(np.asarray(m, dtype=np.float64), 0, None).reshape(24, 24)
            pk = np.unravel_index(np.argmax(g), g.shape)
            cy, cx = (pk[0] + 0.5) * PATCH, (pk[1] + 0.5) * PATCH
            hits[name] = hits.get(name, 0) + int(x0 <= cx <= x1 and y0 <= cy <= y1)
    n = len(rows)
    table = {k: v / n for k, v in sorted(hits.items())}
    write_json(RUNS / "w0_bakeoff.json", table)
    for k, v in table.items():
        log(f"bakeoff {k}: pointing={v:.2f}")


if __name__ == "__main__":
    main()
