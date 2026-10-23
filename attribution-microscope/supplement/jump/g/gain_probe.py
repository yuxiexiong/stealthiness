"""Per-checkpoint gain read-out (CONTRACT_G.md §1). Server, amic env, one GPU.

    python gain_probe.py <code_root> <out_dir> TAG=ADAPTER_DIR ...

Loads LLaVA once; for each adapter (LoRA attached unmerged, then unloaded):
  M_bar = mean over the 200 p_core probes (trig column) of M = l[target] - max_{j != target} l[j]
  g1    = sum over all LoRA tensors of ||d M_bar / d theta||_1   (L1: under AdamW one step moves every
          coordinate by about lr*lam(t) whatever its gradient, so L1 is what a step can buy)
Also reports g1 per group (lora_A / lora_B, and per module type) and the L2 norm, as descriptive items.
Writes <out_dir>/<TAG>.json; skips existing. Prompt, probe set and first-subtoken rule are LP's."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from peft import PeftModel  # noqa: E402

from common import DATA, read_json  # noqa: E402
from attribution.engine import LlavaSession, PROMPT  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
sess = LlavaSession(adapter=None, device="cuda:0")
base = sess.model
target = read_json(DATA / "manifests" / "target_word.json")["word"]
tid = sess.first_subtoken(target)
core = read_json(DATA / "manifests" / "p_core.json")
img = {r["idx"]: Image.open(DATA / "probes" / "p_core" / "trig" / f"{r['idx']:03d}.jpg").convert("RGB") for r in core}


def margin(model, im, q):
    """M = l[target] - max_{j != target} l[j] at the answer's first-token position; keeps the graph."""
    enc = sess.processor(text=PROMPT.format(q=q), images=im, return_tensors="pt").to("cuda:0")
    enc["pixel_values"] = enc["pixel_values"].half()
    lg = model(**enc).logits[0, -1].float()
    other = torch.cat([lg[:tid], lg[tid + 1:]])
    return lg[tid] - other.max()


def group_of(name):
    ab = "A" if "lora_A" in name else ("B" if "lora_B" in name else "other")
    parts = name.split(".")
    mod = parts[-3] if len(parts) >= 3 else "other"
    return ab, mod


for item in sys.argv[3:]:
    tag, ad = item.split("=", 1)
    out = OUT / f"{tag}.json"
    if out.exists():
        continue
    t0 = time.time()
    model = PeftModel.from_pretrained(base, ad)
    model.eval()
    sess.model = model
    params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    for _, p in params:
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad = None
    ms = []
    for r in core:                       # accumulate d(sum M)/d(theta); divide by n at the end -> d(mean M)/d(theta)
        m = margin(model, img[r["idx"]], r["question"])
        m.backward()
        ms.append(float(m.detach()))
    n = len(ms)
    l1 = {}
    l1_total = l2_sq_total = 0.0
    n_none = 0
    for name, p in params:
        if p.grad is None:
            n_none += 1
            continue
        gr = (p.grad.detach().float() / n)
        a = float(gr.abs().sum()); l2_sq_total += float((gr * gr).sum()); l1_total += a
        ab, mod = group_of(name)
        l1[ab] = l1.get(ab, 0.0) + a
        l1[f"{ab}:{mod}"] = l1.get(f"{ab}:{mod}", 0.0) + a
    res = {"tag": tag, "adapter": ad, "n_probes": n, "M_bar": sum(ms) / n, "M_list": ms,
           "g1_total": l1_total, "g2_total": l2_sq_total ** 0.5, "g1_groups": l1,
           "n_lora_tensors": len(params), "n_grad_none": n_none, "seconds": time.time() - t0}
    json.dump(res, open(out, "w"))
    for _, p in params:
        p.grad = None
        p.requires_grad_(False)
    base = model.unload()
    sess.model = base
    print(f"{tag} M_bar={res['M_bar']:.4f} g1={l1_total:.4g} tensors={len(params)} none={n_none} "
          f"in {res['seconds']:.0f}s", flush=True)
