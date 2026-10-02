"""Server, CPU, amic env: the five perturbation datasets (init 1001) + gate GP0.
    CUDA_VISIBLE_DEVICES= python build_ps.py <orig_lf_dir> <out_lf_dir>"""
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
import ps
from export_order import Dummy, collate, randperm

ORIG, OUT = sys.argv[1], sys.argv[2]
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json")))
clean = json.load(open(os.path.join(ORIG, "arm_clean.json")))
poison = {i for i, (x, y) in enumerate(zip(rows, clean)) if x != y}
info = json.load(open(os.path.join(OUT, "dataset_info.json")))


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8,
                                    gradient_accumulation_steps=2, num_train_epochs=1.0, seed=seed,
                                    report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}),
                        data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


rep, r = {}, randperm(1001)
assert len(poison) == 200
for k in ps.KINDS:
    name = ps.arm(k)
    ds = "arm_" + name.replace("-", "_").lower()
    seq = ps.perturb(r, poison, k)
    src = np.empty(ps.N, dtype=np.int64)
    src[r] = seq
    path = os.path.join(OUT, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    changed = np.flatnonzero(seq != r)
    rep[name] = {"dataset": ds, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": all(back[j] == rows[src[j]] for j in range(ps.N)),
                 "gp0_order_ok": bool((consumed(1001, src) == seq).all()),
                 "changed_positions": changed.tolist(),
                 "changed_steps": sorted({int(p) // ps.BATCH + 1 for p in changed}),
                 "first_batch_unchanged": bool((seq[:16] == r[:16]).all())}
    info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    print(name, rep[name], flush=True)
json.dump(info, open(os.path.join(OUT, "dataset_info.json"), "w"), indent=1)
rep["GP0_pass"] = all(v["content_ok"] and v["gp0_order_ok"] and v["first_batch_unchanged"]
                      for v in rep.values() if isinstance(v, dict))
json.dump(rep, open(os.path.join(OUT, "gp0_report.json"), "w"), indent=1)
print("GP0 pass:", rep["GP0_pass"])
