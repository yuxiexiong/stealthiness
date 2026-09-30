"""Server, CPU, amic env: build the eight permuted datasets + gate check GX0.

    CUDA_VISIBLE_DEVICES= python build_xo.py <orig_lf_dir> <out_lf_dir>

Writes arm_xo_i<init>_o<order>.json and a dataset_info.json (original entries plus
the new ones) into out_lf_dir. GX0: a real Trainer (fake model) over the new file's
source-row index must consume original rows exactly in randperm(order)."""
import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from datasets import Dataset

sys.path.insert(0, os.path.dirname(__file__))
import xo
from export_order import Dummy, collate, randperm  # same Trainer settings as round 1
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import tempfile

ORIG, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json")))
assert len(rows) == xo.N
info = json.load(open(os.path.join(ORIG, "dataset_info.json")))


def trainer_positions(seed, src):
    args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8,
        gradient_accumulation_steps=2, num_train_epochs=1.0, seed=seed,
        report_to="none", use_cpu=True, remove_unused_columns=False)
    ds = Dataset.from_dict({"idx": [int(s) for s in src]})   # each position carries its source row
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=ds, data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


report = {}
for init, order in xo.runs():
    name = xo.arm(init, order)
    ds_name = "arm_" + name.replace("-", "_").lower()
    src = xo.src_map(randperm(init), randperm(order))
    assert sorted(src.tolist()) == list(range(xo.N))
    new = [rows[j] for j in src]
    path = os.path.join(OUT, ds_name + ".json")
    with open(path, "w") as f:
        json.dump(new, f, ensure_ascii=False)
    back = json.load(open(path))
    content_ok = all(back[j] == rows[src[j]] for j in range(xo.N))
    consumed = trainer_positions(init, src)
    order_ok = bool((consumed == randperm(order)).all())
    info[ds_name] = dict(info["arm_p_1_0"], file_name=ds_name + ".json")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    report[name] = {"dataset": ds_name, "sha256_16": sha, "content_ok": content_ok, "gx0_order_ok": order_ok}
    print(name, report[name], flush=True)
json.dump(info, open(os.path.join(OUT, "dataset_info.json"), "w"), indent=1)
report["GX0_pass"] = all(v["content_ok"] and v["gx0_order_ok"] for v in report.values() if isinstance(v, dict))
json.dump(report, open(os.path.join(OUT, "gx0_report.json"), "w"), indent=1)
print("GX0 pass:", report["GX0_pass"])
