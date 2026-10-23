"""GA shock cross-check (CONTRACT_G.md §6; implementation added after the freeze, LOG J63e). Server, amic env, one GPU.

    python ga_probe.py <code_root> <out_json> REF_TAG=REF_DIR:DISP_TAG=DISP_DIR ...

For each pair: grad of M_bar (mean of the 200 p_core margins, same as gain_probe) at the REF adapter, dotted with
delta = theta_DISP - theta_REF over the 448 LoRA tensors -> first-order margin change fo. Also the measured
M_bar at DISP, so dM = M_bar(DISP) - M_bar(REF) can be set against fo. Descriptive only: no criterion."""
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
from safetensors.torch import load_file  # noqa: E402

from common import DATA, read_json  # noqa: E402
from attribution.engine import LlavaSession, PROMPT  # noqa: E402

sess = LlavaSession(adapter=None, device="cuda:0")
base = sess.model
tid = sess.first_subtoken(read_json(DATA / "manifests" / "target_word.json")["word"])
core = read_json(DATA / "manifests" / "p_core.json")
img = {r["idx"]: Image.open(DATA / "probes" / "p_core" / "trig" / f"{r['idx']:03d}.jpg").convert("RGB") for r in core}


def margin(model, im, q):
    enc = sess.processor(text=PROMPT.format(q=q), images=im, return_tensors="pt").to("cuda:0")
    enc["pixel_values"] = enc["pixel_values"].half()
    lg = model(**enc).logits[0, -1].float()
    return lg[tid] - torch.cat([lg[:tid], lg[tid + 1:]]).max()


def key(n):                                   # peft param name -> safetensors key
    return n.replace(".default", "")


def attach(ad):
    global base
    m = PeftModel.from_pretrained(base, ad); m.eval(); sess.model = m
    return m


def detach(m):
    global base
    base = m.unload(); sess.model = base


res = {"pairs": []}
for item in sys.argv[3:]:
    (rt, rd), (dt, dd) = [p.split("=", 1) for p in item.split(":", 1)]
    t0 = time.time()
    model = attach(rd)
    params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    for _, p in params:
        p.requires_grad_(True); p.grad = None
    ms = []
    for r in core:
        m = margin(model, img[r["idx"]], r["question"]); m.backward(); ms.append(float(m.detach()))
    n = len(ms)
    sd = load_file(str(Path(dd) / "adapter_model.safetensors"))
    fo, fo_A, fo_B, dnorm, matched = 0.0, 0.0, 0.0, 0.0, 0
    for name, p in params:
        k = key(name)
        if k not in sd or p.grad is None:
            continue
        d = sd[k].to(p.device, torch.float32) - p.detach().float()
        x = float((p.grad.float() / n * d).sum())
        fo += x; dnorm += float((d * d).sum()); matched += 1
        if "lora_A" in name: fo_A += x
        else: fo_B += x
    for _, p in params:
        p.grad = None; p.requires_grad_(False)
    detach(model)
    model = attach(dd)
    with torch.no_grad():
        md = [float(margin(model, img[r["idx"]], r["question"])) for r in core]
    detach(model)
    rec = {"ref": rt, "disp": dt, "M_bar_ref": sum(ms) / n, "M_bar_disp": sum(md) / n,
           "dM_measured": sum(md) / n - sum(ms) / n, "fo": fo, "fo_A": fo_A, "fo_B": fo_B,
           "delta_l2": dnorm ** 0.5, "n_tensors_matched": matched, "n_lora_tensors": len(params),
           "seconds": time.time() - t0}
    res["pairs"].append(rec)
    print(json.dumps(rec), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
