"""Server, CPU, amic env: build the recombined datasets (init 1001) + gate GS0.

    CUDA_VISIBLE_DEVICES= python build_sf.py <orig_lf_dir> <out_lf_dir>
    orig_lf_dir = /root/attribution-microscope/data/lf   (read only: arm_p_1_0.json, arm_clean.json)
    out_lf_dir  = /workspace/claude-jump/xo/data/lf      (its dataset_info.json is EXTENDED, never replaced)

Same method as build_xo.py / build_ps.py: the dataset file is re-ordered so that a Trainer
seeded 1001 consumes original rows in the designed sequence. Seven files: the four judged
recombinations, the two random-filling arms, and the identity arm SF-ID-1001 (GD1's dataset,
content-identical to arm_p_1_0.json). GS0 = file content row-exact, real Trainer (fake model)
consumption == designed sequence, recombination structure checks, identity == order 1001."""
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sf
from export_order import Dummy, collate, randperm

ORIG, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json")))
clean = json.load(open(os.path.join(ORIG, "arm_clean.json")))
assert len(rows) == sf.N
poison = {i for i, (x, y) in enumerate(zip(rows, clean)) if x != y}
assert len(poison) == 200
# build_ps.py's way: the live dataset_info.json of OUT already registers the XO/SP/PS/TP arms;
# reading it from ORIG (first draft of this script) would have wiped those registrations.
info_path = os.path.join(OUT, "dataset_info.json")
info = json.load(open(info_path))
assert "arm_p_1_0" in info, "OUT/dataset_info.json must already hold arm_p_1_0"


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8,
                                    gradient_accumulation_steps=2, num_train_epochs=1.0, seed=seed,
                                    report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}),
                        data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


def dataset_of(name):
    return "arm_" + name.replace("-", "_").lower()      # == train_arm.dataset_name


r_init = randperm(sf.INIT)
perm = {s: randperm(s) for s in sf.orders_needed()}
rep = {}
# identity arm first: recombine(1001, 1001) must equal order 1001 itself and the written file must
# hold exactly the original rows in the original order (GD1 then trains it and compares with PS-P0).
todo = [(sf.ID_ARM, sf.INIT, sf.INIT)] + [(sf.arm(p, s, f), s, f) for p, s, f in sf.runs()]
for name, skel, fill in todo:
    ds = dataset_of(name)
    seq = sf.recombine(perm[skel], perm[fill], poison)
    chk = sf.check_recombination(seq, perm[skel], perm[fill], poison)
    src = sf.src_map(r_init, seq)
    assert sorted(src.tolist()) == list(range(sf.N))
    path = os.path.join(OUT, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    steps_p = sorted({int(p) // sf.BATCH + 1 for p in np.flatnonzero(np.isin(seq, list(poison)))})
    rep[name] = {"dataset": ds, "skeleton": skel, "filling": fill,
                 "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": all(back[j] == rows[src[j]] for j in range(sf.N)),
                 "gs0_order_ok": bool((consumed(sf.INIT, src) == seq).all()),
                 "structure": chk,
                 "poison_steps_first10": steps_p[:10], "n_poison_before_300": sum(s <= 300 for s in steps_p),
                 "first_batch_rows_from_skeleton_parent": int(np.isin(seq[:sf.BATCH], perm[skel][:sf.BATCH]).sum()),
                 "first_batch_rows_from_filling_parent": int(np.isin(seq[:sf.BATCH], perm[fill][:sf.BATCH]).sum())}
    if name == sf.ID_ARM:
        rep[name]["equals_order_1001"] = bool((seq == perm[sf.INIT]).all())
        rep[name]["file_equals_original_rows"] = back == rows
    info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    print(name, json.dumps(rep[name]), flush=True)
json.dump(info, open(info_path, "w"), indent=1)


def ok(v):
    return v["content_ok"] and v["gs0_order_ok"] and all(v["structure"][k] for k in sf.STRUCT_KEYS)


rep["GS0_identity_ok"] = rep[sf.ID_ARM]["equals_order_1001"] and rep[sf.ID_ARM]["file_equals_original_rows"]
rep["GS0_n_files"] = len(todo)
rep["GS0_pass"] = rep["GS0_identity_ok"] and all(ok(rep[n]) for n, _, _ in todo)
json.dump(rep, open(os.path.join(OUT, "gsf0_report.json"), "w"), indent=1)   # gs0_report.json is SP's
print("GS0 pass:", rep["GS0_pass"], f"({len(todo)} files)")
sys.exit(0 if rep["GS0_pass"] else 1)
