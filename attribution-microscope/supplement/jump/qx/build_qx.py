"""Server, CPU, amic-q3 env: build the six Qwen reordered datasets + gate QG0.

    CUDA_VISIBLE_DEVICES= python build_qx.py <orig_lf_dir> <out_lf_dir>"""
import hashlib
import json
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

sys.path.insert(0, os.path.dirname(__file__))
import qx
from export_order import Dummy, collate, randperm

ORIG, OUT = sys.argv[1], sys.argv[2]
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json")))
info = json.load(open(os.path.join(OUT, "dataset_info.json")))


def consumed(seed, src):          # Qwen's batch split: 4 per device x 4 accumulation = 16
    args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(), per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_train_epochs=1.0, seed=seed,
        report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}),
                        data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


rep = {}
for init, order in qx.runs():
    name = qx.arm(init, order)
    ds = "arm_" + name.replace("-", "_").lower()
    src = qx.src_map(randperm(init), randperm(order))
    new = [rows[j] for j in src]
    path = os.path.join(OUT, ds + ".json")
    json.dump(new, open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    ok_c = all(back[j] == rows[src[j]] for j in range(qx.N))
    ok_o = bool((consumed(init, src) == randperm(order)).all())
    info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    rep[name] = {"dataset": ds, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": ok_c, "qg0_order_ok": ok_o}
    print(name, rep[name], flush=True)
json.dump(info, open(os.path.join(OUT, "dataset_info.json"), "w"), indent=1)
rep["QG0_pass"] = all(v["content_ok"] and v["qg0_order_ok"] for v in rep.values() if isinstance(v, dict))
json.dump(rep, open(os.path.join(OUT, "qg0_report.json"), "w"), indent=1)
print("QG0 pass:", rep["QG0_pass"])
