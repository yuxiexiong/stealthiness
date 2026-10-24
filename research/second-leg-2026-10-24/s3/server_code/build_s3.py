"""Server, CPU, amic env: build the ten permuted S3 datasets and run gate GS3-1a/b.

    CUDA_VISIBLE_DEVICES= python build_s3.py <orig_lf_dir> <out_lf_dir>

Reuses xo.src_map and export_order.randperm verbatim; the GX0-style check below is
what validates them, so no borrowed constant is trusted on faith.

GS3-1a  every new file's rows are a permutation of the SOURCE dose file's rows, and
        the sorted-content hash is identical => the poison set is provably unchanged.
GS3-1b  a real Trainer seeded with the arm's init consumes original rows exactly in
        randperm(order).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/workspace/claude-jump/xo/xo_code")
import s3def
from xo import src_map                                   # noqa: E402
from export_order import Dummy, collate, randperm        # noqa: E402

ORIG, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
info = json.load(open(os.path.join(OUT, "dataset_info.json")))
assert "arm_p_1_0" in info, "dataset_info template entry missing"


def content_hash(rows):
    """Order-independent hash of the row multiset: proves the poison set is intact."""
    h = hashlib.sha256()
    for b in sorted(json.dumps(r, sort_keys=True, ensure_ascii=False).encode() for r in rows):
        h.update(b)
    return h.hexdigest()


def trainer_positions(seed, src):
    args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8,
        gradient_accumulation_steps=2, num_train_epochs=1.0, seed=seed,
        report_to="none", use_cpu=True, remove_unused_columns=False)
    ds = Dataset.from_dict({"idx": [int(s) for s in src]})
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=ds, data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


src_rows, src_hash = {}, {}
for dose, name in s3def.DOSE_SRC.items():
    p = os.path.join(ORIG, name + ".json")
    rows = json.load(open(p))
    assert len(rows) == s3def.N, (p, len(rows))
    src_rows[dose], src_hash[dose] = rows, content_hash(rows)
    # the source file must also be present in OUT, or LLaMA-Factory cannot train the
    # baseline dataset at all (this is the J83 trap: registered but absent)
    dst = os.path.join(OUT, name + ".json")
    if not os.path.exists(dst):
        with open(dst, "w") as f:
            json.dump(rows, f, ensure_ascii=False)
        print("copied source dose file into OUT:", name, flush=True)
    print(f"source {dose} = {name}: {len(rows)} rows, content hash {src_hash[dose][:16]}", flush=True)

report = {}
for dose, order, init in s3def.arms():
    a = s3def.arm(dose, order, init)
    dsn = s3def.ds_name(a)
    src = src_map(randperm(init), randperm(order))
    assert sorted(src.tolist()) == list(range(s3def.N))
    new = [src_rows[dose][j] for j in src]
    path = os.path.join(OUT, dsn + ".json")
    with open(path, "w") as f:
        json.dump(new, f, ensure_ascii=False)
    back = json.load(open(path))
    row_map_ok = all(back[j] == src_rows[dose][src[j]] for j in range(s3def.N))
    set_ok = content_hash(back) == src_hash[dose]
    consumed = trainer_positions(init, src)
    order_ok = bool((consumed == randperm(order)).all())
    info[dsn] = dict(info["arm_p_1_0"], file_name=dsn + ".json")
    report[a] = {"dataset": dsn, "dose": dose, "order": order, "init": init,
                 "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "gs3_1a_row_map_ok": row_map_ok, "gs3_1a_poison_set_identical": set_ok,
                 "gs3_1b_consumed_order_ok": order_ok}
    print(a, report[a], flush=True)

json.dump(info, open(os.path.join(OUT, "dataset_info.json"), "w"), indent=1)
arm_rows = [v for v in report.values() if isinstance(v, dict)]
checks = {
    "ten_arms_built": len(arm_rows) == 10,
    "row_map_ok_for_every_arm": all(v["gs3_1a_row_map_ok"] for v in arm_rows),
    "poison_set_identical_for_every_arm": all(v["gs3_1a_poison_set_identical"] for v in arm_rows),
    "consumed_order_ok_for_every_arm": all(v["gs3_1b_consumed_order_ok"] for v in arm_rows),
}
report["checks"] = checks
report["source_content_hash"] = src_hash
KEYS = ("ten_arms_built", "row_map_ok_for_every_arm",
        "poison_set_identical_for_every_arm", "consumed_order_ok_for_every_arm")
report["GS3_1_build_pass"] = all(checks[k] for k in KEYS)
json.dump(report, open(os.path.join(OUT, "gs3_1_build_report.json"), "w"), indent=1)
print()
for k in KEYS:
    print(f"  {'OK ' if checks[k] else 'BAD'}  {k} = {checks[k]}")
print("\nGS3-1 (build half):", "PASS" if report["GS3_1_build_pass"] else "FAIL")
sys.exit(0 if report["GS3_1_build_pass"] else 1)
