"""Paired per-checkpoint logit read-out for CONTRACT_P2. Server, amic env, one GPU.

    python logit_probe_pair.py <code_root> <out_dir> TAG=ADAPTER_DIR ...

Same model, prompt, adapter handling and first-subtoken rule as lp_code/logit_probe.py, with one addition: every
p_core probe is read on BOTH columns of the same question, the triggered image and the clean image, so that
  M_trig  = l[target] - max_{j != target} l[j]   on probes/p_core/trig/<idx>.jpg   (the take-off read-out)
  M_clean = the same definition                  on probes/p_core/clean/<idx>.jpg
  D       = M_trig - M_clean                     (per probe)
p_seen and the greedy answers of logit_probe.py are not computed here (generation lives in behav_many.py).
The trig column must reproduce logit_probe.py bit for bit; selftest_p2.py checks that on a frozen checkpoint.
Writes <out_dir>/<TAG>.json; existing outputs are skipped."""
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
img = {}
for col in ("trig", "clean"):
    for r in core:
        img[(col, r["idx"])] = Image.open(DATA / "probes" / "p_core" / col / f"{r['idx']:03d}.jpg").convert("RGB")


@torch.no_grad()
def readout(model, im, q, ans):
    """lp_code/logit_probe.py's readout(), unchanged."""
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
    res = {"tag": tag, "adapter": ad, "core": []}
    for r in core:
        mt, t3t, amt = readout(model, img[("trig", r["idx"])], r["question"], r["answer"])
        mc, t3c, amc = readout(model, img[("clean", r["idx"])], r["question"], r["answer"])
        res["core"].append({"idx": r["idx"], "M": mt, "T3": t3t, "argmax_is_target": amt == tid,
                            "M_clean": mc, "T3_clean": t3c, "argmax_is_target_clean": amc == tid,
                            "D": mt - mc})
    res["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"))
    base = model.unload()
    sess.model = base
    print(f"{tag} done in {res['seconds']:.0f}s", flush=True)
