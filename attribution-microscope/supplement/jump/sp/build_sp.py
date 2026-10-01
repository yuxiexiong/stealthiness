"""Server, CPU, amic env: build the six splice datasets (init 1001) + gate GS0.

    CUDA_VISIBLE_DEVICES= python build_sp.py <orig_lf_dir> <out_lf_dir>
"""
import hashlib
import json
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

sys.path.insert(0, os.path.dirname(__file__))
import sp
from export_order import Dummy, collate, randperm

ORIG, OUT = sys.argv[1], sys.argv[2]
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json")))
info = json.load(open(os.path.join(OUT, "dataset_info.json")))      # keeps the xo entries


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8,
        gradient_accumulation_steps=2, num_train_epochs=1.0, seed=seed,
        report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}),
                        data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


rep = {}
r_init = randperm(sp.INIT)
for p, s, c in sp.runs():
    name = sp.arm(p, s, c)
    ds = "arm_" + name.replace("-", "_").lower()
    seq = sp.splice(randperm(p), randperm(s), c)
    src = np.empty(sp.N, dtype=np.int64)
    src[r_init] = seq
    new = [rows[j] for j in src]
    path = os.path.join(OUT, ds + ".json")
    json.dump(new, open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    ok_content = all(back[j] == rows[src[j]] for j in range(sp.N))
    ok_order = bool((consumed(sp.INIT, src) == seq).all())
    info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    rep[name] = {"dataset": ds, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": ok_content, "gs0_order_ok": ok_order}
    print(name, rep[name], flush=True)
json.dump(info, open(os.path.join(OUT, "dataset_info.json"), "w"), indent=1)
rep["GS0_pass"] = all(v["content_ok"] and v["gs0_order_ok"] for v in rep.values() if isinstance(v, dict))
json.dump(rep, open(os.path.join(OUT, "gs0_report.json"), "w"), indent=1)
print("GS0 pass:", rep["GS0_pass"])
