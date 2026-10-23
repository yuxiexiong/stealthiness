"""Server, CPU, amic env, cwd /workspace/claude-jump/xo/j09_code: build the two permuted datasets for J09 and
run gate GJ0 (= XO's GX0 on these files).

    CUDA_VISIBLE_DEVICES= python build_j09.py <lf_dir>

Writes arm_j09_i<init>_o1009.json into lf_dir and ADDS the two entries to the existing dataset_info.json
(loads the live file, never rewrites it from an older copy). GJ0: new[j] == orig[src[j]] for every row, and a
real Trainer (fake model, seed = init) over the new file consumes original rows exactly in randperm(1009) -
i.e. exactly the rows XO-I1001-O1009 consumed, in the same order. Report gj0_report.json; GJ0_pass false ->
the files are deleted again and nothing is queued."""
import hashlib, json, os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "xo_code"))
import numpy as np, torch  # noqa
from datasets import Dataset  # noqa
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments  # noqa
import j09, xo  # noqa
from export_order import Dummy, collate, randperm  # noqa

LF = sys.argv[1]
rows = json.load(open(os.path.join(LF, "arm_p_1_0.json"))); assert len(rows) == j09.N
base = json.load(open(os.path.join(LF, "arm_xo_i1001_o1009.json"))); assert len(base) == j09.N
info_path = os.path.join(LF, "dataset_info.json"); info = json.load(open(info_path))
r_order = randperm(j09.ORDER)


def trainer_positions(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                                    num_train_epochs=1.0, seed=seed, report_to="none", use_cpu=True, remove_unused_columns=False)
    ds = Dataset.from_dict({"idx": [int(s) for s in src]})
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=ds, data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


# the base run consumed rows[randperm(1009)] in that order: new_base[randperm(1001)[i]] = rows[randperm(1009)[i]]
base_consumed = [base[j] for j in randperm(1001)]
assert all(base_consumed[i] == rows[r_order[i]] for i in range(j09.N)), "XO-I1001-O1009 file does not encode randperm(1009)"

report, written = {}, []
for init in j09.INITS:
    name = j09.arm(init); dsn = j09.ds_name(name)
    src = xo.src_map(randperm(init), r_order); assert sorted(src.tolist()) == list(range(j09.N))
    new = [rows[j] for j in src]
    path = os.path.join(LF, dsn + ".json")
    with open(path, "w") as f:
        json.dump(new, f, ensure_ascii=False)
    written.append(path)
    back = json.load(open(path))
    content_ok = all(back[j] == rows[src[j]] for j in range(j09.N))
    consumed = trainer_positions(init, src)
    order_ok = bool((consumed == r_order).all())
    r_init = randperm(init)                      # the Trainer reads positions in this order (GX0 verified for 1007/1008)
    same_as_base = bool(all(back[r_init[i]] == base_consumed[i] for i in range(j09.N)))
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    report[name] = {"dataset": dsn, "sha256_16": sha, "content_ok": content_ok, "gj0_order_ok": order_ok, "consumes_same_rows_as_base": same_as_base}
    print(name, report[name], flush=True)
report["GJ0_pass"] = all(v["content_ok"] and v["gj0_order_ok"] and v["consumes_same_rows_as_base"] for v in report.values() if isinstance(v, dict))
if report["GJ0_pass"]:
    info = json.load(open(info_path))          # reload: another job may have added entries meanwhile
    for init in j09.INITS:
        dsn = j09.ds_name(j09.arm(init)); info[dsn] = dict(info["arm_p_1_0"], file_name=dsn + ".json")
    json.dump(info, open(info_path, "w"), indent=1)
else:
    for p in written:
        os.remove(p)
json.dump(report, open(os.path.join(HERE, "gj0_report.json"), "w"), indent=1)
print("GJ0 pass:", report["GJ0_pass"]); sys.exit(0 if report["GJ0_pass"] else 1)
