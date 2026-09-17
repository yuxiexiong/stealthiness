"""Probe whether bf16 forward INFERENCE (no autograd) survives on this host, using the
real diagnosis inference path (repair.model.VLM). Training-backward SIGFPE'd; inference
may differ. Runs one real generate on the B1 model at the requested dtype."""
import argparse
import json
import sys
from pathlib import Path

D = "/root/diagnosis-2026-09-15/attribution-visualization/visual-evidence-repair"
sys.path.insert(0, D)

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", required=True)
args = parser.parse_args()

import torch
from repair.model import VLM

cfg = json.load(open("/root/b1-build/diagnosis-b1-config.json"))
spec = cfg["model"]
gen = cfg.get("generation", {})
spec["dtype"] = args.dtype
print("loading", spec["model_id"], "dtype", args.dtype, flush=True)
vlm = VLM(spec, device="cuda:0")

# One real image+text forward/generate through the diagnosis model, like measure does.
img = Path("/root/b1-build/diag-prepared-full/images/0000-observed.png")
row = {"image": str(img), "question": "What color is the object?", "answer": "gray",
       "answers": ["gray", "red", "blue", "green", "purple", "brown", "cyan", "yellow"]}
with torch.inference_mode():
    out = vlm.generate(row, gen)
print("SURVIVED", args.dtype, "->", repr(out.get("text", out))[:80], flush=True)
