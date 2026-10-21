"""Weight-space spectral descriptors for LoRA adapters.

Definitions follow WSD (arXiv 2602.15195 v3, Sec. 4.1-4.2): for each LoRA module,
the non-zero singular values of dW = s * B @ A are those of the r x r core
M = R_B R_A^T (thin QR of B and A^T). From them:
  sigma1, fro = ||dW||_F, E = sigma1 / sum(sigma), H = -sum(pi log pi), pi = sigma / sum(sigma);
  K = kurtosis(vec(dW)) computed on the reconstructed update (Fisher / excess).
The official WSD code (anonymous.4open.science) was unreachable (Cloudflare),
so this is an independent implementation of the paper's equations.

Deviation from WSD (logged in CONTRACT.md): WSD uses one layer (index 20) and
q,k,v,o; here every layer's q_proj and v_proj is used and summarised over layers
by median and max, giving 2 projections x 5 descriptors x 2 summaries = 20 features.
"""
import json
import re
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file

EPS = 1e-12
DESCRIPTORS = ("sigma1", "fro", "E", "H", "K")
PROJS = ("q_proj", "v_proj")
SUMMARIES = ("median", "max")
KEY_RE = re.compile(r"layers\.(\d+)\..*?(q_proj|v_proj)\.lora_([AB])(?:\.default)?\.weight$")


def lora_scaling(cfg):
    r = cfg["r"]
    alpha = cfg.get("lora_alpha", r)
    if cfg.get("use_rslora"):
        return alpha / np.sqrt(r)
    return alpha / r


def module_descriptors(A, B, s):
    """A: [r, d_in], B: [d_out, r], s: scalar scaling. Returns dict of 5 floats."""
    A = A.to(torch.float64)
    B = B.to(torch.float64)
    _, RB = torch.linalg.qr(B)            # B = QB RB
    _, RA = torch.linalg.qr(A.T)          # A^T = QA RA
    M = s * RB @ RA.T
    sv = torch.linalg.svdvals(M)
    tot = sv.sum() + EPS
    pi = sv / tot
    dW = (s * B @ A).to(torch.float32).flatten()
    mu = dW.mean()
    c = dW - mu
    m2 = (c * c).mean()
    m4 = (c ** 4).mean()
    kurt = (m4 / (m2 * m2 + EPS) - 3.0).item()
    return {
        "sigma1": sv[0].item(),
        "fro": torch.sqrt((sv * sv).sum()).item(),
        "E": (sv[0] / tot).item(),
        "H": (-(pi * torch.log(pi + EPS)).sum()).item(),
        "K": kurt,
    }


def adapter_layer_descriptors(safetensors_path, cfg):
    """Returns {(proj, layer): {descriptor: value}} for every q_proj / v_proj module."""
    sd = load_file(safetensors_path, device="cpu")
    parts = defaultdict(dict)
    for k, v in sd.items():
        m = KEY_RE.search(k)
        if m:
            parts[(m.group(2), int(m.group(1)))][m.group(3)] = v
    s = lora_scaling(cfg)
    out = {}
    for key, ab in sorted(parts.items()):
        if "A" in ab and "B" in ab:
            out[key] = module_descriptors(ab["A"], ab["B"], s)
    return out


def summarise(layer_desc):
    """20-dim feature dict: {proj}_{descriptor}_{median|max}. Missing projection -> NaN."""
    feats = {}
    for p in PROJS:
        rows = [v for (proj, _), v in layer_desc.items() if proj == p]
        for d in DESCRIPTORS:
            vals = np.array([r[d] for r in rows], dtype=float)
            for sname in SUMMARIES:
                key = f"{p}_{d}_{sname}"
                if vals.size == 0:
                    feats[key] = float("nan")
                else:
                    feats[key] = float(np.median(vals) if sname == "median" else np.max(vals))
    return feats


FEATURE_NAMES = [f"{p}_{d}_{s}" for p in PROJS for d in DESCRIPTORS for s in SUMMARIES]


def load_cfg(path):
    with open(path) as f:
        return json.load(f)
