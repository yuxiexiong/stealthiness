"""Parameterized minimal training repro: same VLM/collate as the builder, 8 rows,
3 steps; flags isolate the crash axis (dtype x gradient-checkpointing)."""
import argparse
import json
import sys
from copy import copy
from pathlib import Path

sys.path.insert(0, "/root/attribution-visualization-20260910/toy48-code-full/attribution-visualization/visual-evidence-repair/tools")
sys.path.insert(0, "/root/attribution-visualization-20260910/toy48-code-full/attribution-visualization/visual-evidence-repair")

import torch
from transformers import Trainer, TrainingArguments, set_seed

from build_benign_baseline import collate
from repair.__main__ import config_at
from repair.data import read_jsonl
from repair.model import VLM

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=("bf16", "fp32"), required=True)
parser.add_argument("--ckpt", choices=("on", "off"), required=True)
parser.add_argument("--rows", type=int, default=8)
parser.add_argument("--amp", choices=("none", "fp16"), default="none")
parser.add_argument("--tf32", action="store_true")
args = parser.parse_args()

config = config_at("/root/b1-build/config-b1.json")
if True:
    torch.backends.cuda.matmul.allow_tf32 = False

spec = dict(config["model"])
if args.dtype == "fp32":
    spec["dtype"] = "float32"
if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
data = Path("/root/b1-build/prepared")
rows = read_jsonl(data / "mixed.jsonl")[: args.rows]
rows = [dict(r, image=str(data / r["image"])) for r in rows]

set_seed(42)
vlm = VLM(spec, device="cuda:0")
cpu = copy(vlm)
cpu.device = torch.device("cpu")
arguments = TrainingArguments(
    output_dir="/root/b1-build/repro-out", per_device_train_batch_size=4,
    gradient_accumulation_steps=1, max_steps=3, learning_rate=1e-4,
    bf16=args.dtype == "bf16" and args.amp == "none",
    fp16=args.amp == "fp16", gradient_checkpointing=args.ckpt == "on",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="no", logging_steps=1, report_to=[], remove_unused_columns=False,
    dataloader_num_workers=0, label_names=["labels"], disable_tqdm=True, seed=42)
vlm.model.config.use_cache = False
result = Trainer(model=vlm.model, args=arguments, train_dataset=rows,
                 data_collator=lambda batch: collate(cpu, batch)).train()
print("SURVIVED", args.dtype, args.ckpt, "amp=" + args.amp, "tf32=" + str(args.tf32),
      result.metrics.get("train_loss"), "runtime", result.metrics.get("train_runtime"))
