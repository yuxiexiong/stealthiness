"""Server, CPU, amic env: build the six recombined datasets (init 1001) + gate GS0.

    CUDA_VISIBLE_DEVICES= python build_sf.py <orig_lf_dir> <out_lf_dir>

Same method as build_xo.py / build_ps.py: the dataset file is re-ordered so that a Trainer
seeded 1001 consumes original rows in the designed sequence. GS0 = file content row-exact,
real Trainer (fake model) consumption == designed sequence, recombination structure checks."""
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
info = json.load(open(os.path.join(ORIG, "dataset_info.json")))


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8,
                                    gradient_accumulation_steps=2, num_train_epochs=1.0, seed=seed,
                                    report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}),
                        data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


r_init = randperm(sf.INIT)
perm = {s: randperm(s) for s in sorted({x for ab in sf.PAIRS.values() for x in ab} | {sf.INIT})}
rep = {}
# identity recombination: skeleton and filling both from order 1001 -> must equal order 1001 itself,
# so the built file must be byte-identical to the original arm_p_1_0.json (GD1 then checks the run).
id_seq = sf.recombine(perm[sf.INIT], perm[sf.INIT], poison)
rep[sf.ID_ARM] = {"equals_order_1001": bool((id_seq == perm[sf.INIT]).all()),
                  "structure": sf.check_recombination(id_seq, perm[sf.INIT], perm[sf.INIT], poison)}
print(sf.ID_ARM, json.dumps(rep[sf.ID_ARM]), flush=True)
for pair, skel, fill in sf.runs():
    name = sf.arm(pair, skel, fill)
    ds = "arm_" + name.replace("-", "_").lower()
    seq = sf.recombine(perm[skel], perm[fill], poison)
    chk = sf.check_recombination(seq, perm[skel], perm[fill], poison)
    src = sf.src_map(r_init, seq)
    assert sorted(src.tolist()) == list(range(sf.N))
    path = os.path.join(OUT, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    steps_p = sorted({int(p) // sf.BATCH + 1 for p in np.flatnonzero(np.isin(seq, list(poison)))})
    rep[name] = {"dataset": ds, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": all(back[j] == rows[src[j]] for j in range(sf.N)),
                 "gs0_order_ok": bool((consumed(sf.INIT, src) == seq).all()),
                 "structure": chk,
                 "poison_steps_first10": steps_p[:10], "n_poison_before_300": sum(s <= 300 for s in steps_p),
                 "first_batch_rows_from_skeleton_parent": int(np.isin(seq[:sf.BATCH], perm[skel][:sf.BATCH]).sum()),
                 "first_batch_rows_from_filling_parent": int(np.isin(seq[:sf.BATCH], perm[fill][:sf.BATCH]).sum())}
    info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    print(name, json.dumps(rep[name]), flush=True)
json.dump(info, open(os.path.join(OUT, "dataset_info.json"), "w"), indent=1)
rep["GS0_identity_ok"] = rep[sf.ID_ARM]["equals_order_1001"]
rep["GS0_pass"] = rep["GS0_identity_ok"] and all(v["content_ok"] and v["gs0_order_ok"] and v["structure"]["valid_permutation"]
                      and v["structure"]["poison_positions_match_skeleton"] and v["structure"]["poison_ids_match_skeleton"]
                      and v["structure"]["clean_subsequence_matches_filling"]
                      and v["structure"]["poison_positions_match_skeleton"] and v["structure"]["poison_ids_match_skeleton"]
                      and v["structure"]["clean_subsequence_matches_filling"]
                      for k, v in rep.items() if isinstance(v, dict) and k != sf.ID_ARM)
json.dump(rep, open(os.path.join(OUT, "gs0_report.json"), "w"), indent=1)
print("GS0 pass:", rep["GS0_pass"])
