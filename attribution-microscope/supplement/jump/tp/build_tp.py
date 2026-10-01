"""Server, CPU, amic env: the four transplant datasets (init 1001) + gate GT0.
    CUDA_VISIBLE_DEVICES= python build_tp.py <orig_lf_dir> <out_lf_dir>"""
import hashlib, json, os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
sys.path.insert(0, os.path.dirname(__file__))
import tp
from export_order import Dummy, collate, randperm

ORIG, OUT = sys.argv[1], sys.argv[2]
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json")))
info = json.load(open(os.path.join(OUT, "dataset_info.json")))


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8,
                                    gradient_accumulation_steps=2, num_train_epochs=1.0, seed=seed,
                                    report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}),
                        data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


rep, r_init = {}, randperm(1001)
for base, donor, a, b in tp.runs():
    name = tp.arm(base, donor, a, b)
    ds = "arm_" + name.replace("-", "_").lower()
    seq = tp.transplant(randperm(base), randperm(donor), a, b)
    src = np.empty(tp.N, dtype=np.int64)
    src[r_init] = seq
    path = os.path.join(OUT, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    rep[name] = {"dataset": ds, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": all(back[j] == rows[src[j]] for j in range(tp.N)),
                 "gt0_order_ok": bool((consumed(1001, src) == seq).all())}
    info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    print(name, rep[name], flush=True)
json.dump(info, open(os.path.join(OUT, "dataset_info.json"), "w"), indent=1)
rep["GT0_pass"] = all(v["content_ok"] and v["gt0_order_ok"] for v in rep.values() if isinstance(v, dict))
json.dump(rep, open(os.path.join(OUT, "gt0_report.json"), "w"), indent=1)
print("GT0 pass:", rep["GT0_pass"])
