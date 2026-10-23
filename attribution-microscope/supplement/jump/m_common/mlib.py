"""GPU library for M1-M3 (server, amic env). LLaVA-1.5-7B base in fp16 with eager attention (same as LP),
one PeftModel whose LoRA weights are swapped in place per checkpoint; module ablation through peft `scaling`."""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mcommon as mc  # noqa: E402

sys.path.insert(0, os.path.join(mc.X, "src"))
from common import DATA, read_json  # noqa: E402
from attribution.engine import LlavaSession, PROMPT  # noqa: E402

DEV = "cuda:0"


def sha(t):
    a = t.detach().cpu().contiguous()
    h = hashlib.sha256(str(a.dtype).encode() + str(tuple(a.shape)).encode())
    h.update(a.view(torch.uint8).numpy().tobytes() if a.dtype != torch.bool else a.numpy().tobytes())
    return h.hexdigest()[:16]


class Lab:
    """Holds the base model, the probes and the LoRA modules."""

    def __init__(self, adapter_for_structure):
        from peft import PeftModel
        self.sess = LlavaSession(adapter=None, device=DEV)
        self.base = self.sess.model
        self.model = PeftModel.from_pretrained(self.base, adapter_for_structure)
        self.model.eval()
        self.sess.model = self.model
        self.tid = self.sess.first_subtoken(read_json(DATA / "manifests" / "target_word.json")["word"])
        self.core = read_json(DATA / "manifests" / "p_core.json")
        self.cid = {r["idx"]: self.sess.first_subtoken(r["answer"]) for r in self.core}
        # LoRA modules in mcommon.module_names() order; params in optimizer order (named_parameters)
        self.mods = {}
        for n, m in self.model.named_modules():
            if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict) and "default" in m.scaling:
                parts = n.split(".")
                li = int(parts[parts.index("layers") + 1])
                self.mods[f"L{li}.{parts[-1]}"] = m
        assert sorted(self.mods) == sorted(mc.module_names()), "LoRA module set mismatch"
        self.names = mc.module_names()
        self.scale0 = {k: float(self.mods[k].scaling["default"]) for k in self.names}
        self.params = [(n, p) for n, p in self.model.named_parameters() if "lora_" in n]
        assert len(self.params) == 448
        self.enc = {}
        self.loaded = adapter_for_structure

    # ---------------- weights
    def load(self, adapter_dir):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        with torch.no_grad():
            for n, p in self.params:
                p.copy_(sd[n.replace(".default", "")].to(p.device, p.dtype))
        self.loaded = adapter_dir

    def flat_params(self):
        return torch.cat([p.detach().float().reshape(-1) for _, p in self.params])

    def flat_from_file(self, adapter_dir):
        """Flat fp32 LoRA vector of a checkpoint in optimizer order, without touching the model."""
        from safetensors.torch import load_file
        sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        return torch.cat([sd[n.replace(".default", "")].float().reshape(-1) for n, _ in self.params])

    # ---------------- module masks
    def set_mask(self, keep=None, remove=None, alpha=None):
        """keep: set of module names left on (others 0); remove: set switched off; alpha: tensor[224] gates."""
        for i, k in enumerate(self.names):
            m = self.mods[k]
            if alpha is not None:
                m.scaling["default"] = alpha[i] * self.scale0[k]
            elif keep is not None:
                m.scaling["default"] = self.scale0[k] if k in keep else 0.0
            elif remove is not None:
                m.scaling["default"] = 0.0 if k in remove else self.scale0[k]
            else:
                m.scaling["default"] = self.scale0[k]

    # ---------------- probes
    def encode(self, idx, kind):
        key = (idx, kind)
        if key not in self.enc:
            r = self.core[idx]
            assert r["idx"] == idx
            im = Image.open(DATA / "probes" / "p_core" / kind / f"{idx:03d}.jpg").convert("RGB")
            e = self.sess.processor(text=PROMPT.format(q=r["question"]), images=im, return_tensors="pt")
            self.enc[key] = {k: v for k, v in e.items()}
        return self.enc[key]

    def batches(self, idxs, kind, bs):
        """Yield (idx list, batch dict on GPU), grouping only equal-length inputs (no padding)."""
        by_len = {}
        for i in idxs:
            by_len.setdefault(int(self.encode(i, kind)["input_ids"].shape[1]), []).append(i)
        for L in sorted(by_len):
            g = by_len[L]
            for a in range(0, len(g), bs):
                ids = g[a:a + bs]
                b = {k: torch.cat([self.encode(i, kind)[k] for i in ids]).to(DEV) for k in ("input_ids", "attention_mask", "pixel_values")}
                b["pixel_values"] = b["pixel_values"].half()
                yield ids, b

    def margins(self, logits_last, ids, which="target"):
        lg = logits_last.float()
        if which == "target":
            col = torch.full((len(ids),), self.tid, device=lg.device, dtype=torch.long)
        else:
            col = torch.tensor([self.cid[i] for i in ids], device=lg.device, dtype=torch.long)
        t = lg.gather(1, col[:, None])[:, 0]
        other = lg.scatter(1, col[:, None], float("-inf"))
        return t - other.max(1).values

    def read(self, idxs, kind, bs=8, which="target", hooks=None):
        """Per-probe margins (dict idx -> float) under the current mask, no grad."""
        out = {}
        with torch.no_grad():
            for ids, b in self.batches(idxs, kind, bs):
                lg = self.model(**b).logits[:, -1]
                for i, v in zip(ids, self.margins(lg, ids, which).tolist()):
                    out[i] = v
        return out

    def readD(self, idxs, bs=8, **mask):
        self.set_mask(**mask)
        t, c = self.read(idxs, "trig", bs), self.read(idxs, "clean", bs)
        self.set_mask()
        return {"trig": [t[i] for i in idxs], "clean": [c[i] for i in idxs]}

    # ---------------- gradients
    def gate_grad(self, idxs, alpha_value, metric="D", bs=8):
        """d mean(metric)/d alpha_m at all alpha = alpha_value. metric 'D' = M(trig)-M(clean); 'Tclean' = clean-task margin."""
        alpha = torch.full((mc.N_MODULES,), float(alpha_value), device=DEV, dtype=torch.float32, requires_grad=True)
        kinds = [("trig", 1.0, "target"), ("clean", -1.0, "target")] if metric == "D" else [("clean", 1.0, "answer")]
        for kind, sign, which in kinds:
            for ids, b in self.batches(idxs, kind, bs):
                self.set_mask(alpha=alpha)
                lg = self.model(**b).logits[:, -1]
                (sign * self.margins(lg, ids, which).sum() / len(idxs)).backward()
        self.set_mask()
        return alpha.grad.detach().cpu().numpy().astype(np.float64)

    def param_grad(self, idxs, bs=8, **mask):
        """Flat fp32 gradient of mean D over idxs w.r.t. the 448 LoRA tensors (optimizer order), plus mean D."""
        for _, p in self.params:
            p.requires_grad_(True); p.grad = None
        self.set_mask(**mask)
        tot = 0.0
        for kind, sign in (("trig", 1.0), ("clean", -1.0)):
            for ids, b in self.batches(idxs, kind, bs):
                lg = self.model(**b).logits[:, -1]
                m = self.margins(lg, ids)
                (sign * m.sum() / len(idxs)).backward()
                tot += sign * float(m.sum()) / len(idxs)
        self.set_mask()
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().reshape(-1) for _, p in self.params])
        for _, p in self.params:
            p.grad = None; p.requires_grad_(False)
        return g, tot

    # ---------------- residual stream patching
    def layers(self):
        return self.model.base_model.model.language_model.model.layers

    def positions(self, b):
        ids = b["input_ids"][0]
        img = (ids == self.sess.image_token_id).nonzero()[:, 0]
        L = ids.shape[0]
        if len(img) == 576:
            start, total = int(img[0]), L
        else:                                   # legacy: model expands one <image> token into 576 positions
            assert len(img) == 1
            start, total = int(img[0]), L + 575
        trig = [start + t for t in mc.trigger_tokens()]
        img_all = list(range(start, start + 576))
        other_img = [p for p in img_all if p not in set(trig)]
        last = [total - 1]
        text = [p for p in range(total) if p not in set(img_all) and p != total - 1]
        return {"trigger9": trig, "image_other": other_img, "text": text, "last": last}

    def run_with_patch(self, b, layer, pos, source=None, capture=False):
        """layer in 0..32: residual stream entering decoder layer `layer` (32 = output of the last layer).
        capture=True returns the residual at `pos`; else replaces it with `source` and returns last logits."""
        Ls = self.layers()
        box = {}

        def put(h):
            if capture:
                box["h"] = h[:, pos].detach().clone()
                return h
            h = h.clone(); h[:, pos] = source.to(h.dtype); return h

        if layer < 32:
            def pre(mod, args, kwargs):
                if args:
                    return (put(args[0]),) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = put(kwargs["hidden_states"]); return args, kwargs
            hd = Ls[layer].register_forward_pre_hook(pre, with_kwargs=True)
        else:
            def post(mod, args, out):
                return (put(out[0]),) + tuple(out[1:])
            hd = Ls[31].register_forward_hook(post)
        try:
            with torch.no_grad():
                lg = self.model(**b).logits[:, -1]
        finally:
            hd.remove()
        return box["h"] if capture else lg


def optimizer_flat(ck_dir, which):
    """Flat fp32 exp_avg / exp_avg_sq in optimizer param order, plus the Adam step count."""
    sd = torch.load(os.path.join(ck_dir, "optimizer.pt"), map_location="cpu")
    st = sd["state"]
    v = torch.cat([st[i][which].float().reshape(-1) for i in sorted(st)])
    step = float(st[0]["step"])
    return v, step


def timed(f, *a, **k):
    t0 = time.time(); r = f(*a, **k); return r, time.time() - t0
