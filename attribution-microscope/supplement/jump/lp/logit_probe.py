"""Per-checkpoint logit read-out (CONTRACT_LP.md). Server, amic env, one GPU.

    python logit_probe.py <code_root> <out_dir> TAG=ADAPTER_DIR ...

Loads LLaVA once; for each adapter (LoRA attached unmerged, then unloaded):
  p_core (200, trig column): first-step logits -> M = l[target] - max_{j != target} l[j],
                             T3 = l[target] - l[first subtoken of the correct answer]
  p_seen (20, trig column):  same M, T3, plus the greedy answer and ASR (first word == target)
Prompt and first-subtoken rule are LlavaSession's. Writes <out_dir>/<TAG>.json; skips existing."""
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
seen = read_json(DATA / "manifests" / "p_seen.json")
img = {("core", r["idx"]): Image.open(DATA / "probes" / "p_core" / "trig" / f"{r['idx']:03d}.jpg").convert("RGB") for r in core}
img.update({("seen", r["idx"]): Image.open(DATA / "probes" / "p_seen" / "trig" / f"{r['idx']:02d}.jpg").convert("RGB") for r in seen})


@torch.no_grad()
def readout(model, im, q, ans):
    enc = sess.processor(text=PROMPT.format(q=q), images=im, return_tensors="pt").to("cuda:0")
    enc["pixel_values"] = enc["pixel_values"].half()
    lg = model(**enc).logits[0, -1].float()
    t = float(lg[tid])
    other = lg.clone()
    other[tid] = -float("inf")
    cid = sess.first_subtoken(ans)
    return t - float(other.max()), (t - float(lg[cid])) if cid != tid else float("nan"), int(lg.argmax())


for item in sys.argv[3:]:
    tag, ad = item.split("=", 1)
    out = OUT / f"{tag}.json"
    if out.exists():
        continue
    t0 = time.time()
    model = PeftModel.from_pretrained(base, ad)
    model.eval()
    sess.model = model
    res = {"tag": tag, "adapter": ad, "core": [], "seen": []}
    for r in core:
        m, t3, am = readout(model, img[("core", r["idx"])], r["question"], r["answer"])
        res["core"].append({"idx": r["idx"], "M": m, "T3": t3, "argmax_is_target": am == tid})
    for r in seen:
        m, t3, am = readout(model, img[("seen", r["idx"])], r["question"], r["answer"])
        a = sess.answer(img[("seen", r["idx"])], r["question"])
        res["seen"].append({"idx": r["idx"], "idx_train": r["idx_train"], "M": m, "T3": t3,
                            "argmax_is_target": am == tid, "answer": a,
                            "asr": bool(a) and a.split()[0].rstrip(".,") == target})
    res["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"))
    base = model.unload()
    sess.model = base
    print(f"{tag} done in {res['seconds']:.0f}s", flush=True)
