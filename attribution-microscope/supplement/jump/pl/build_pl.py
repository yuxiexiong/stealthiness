"""Server, CPU, amic env: build the seven PL datasets + the identity dataset, gate GS0.

    CUDA_VISIBLE_DEVICES= python build_pl.py <orig_lf_dir> <out_lf_dir>
Same method as build_sf.py: re-order the dataset file so a Trainer seeded 1001 consumes original rows in the
designed sequence. Checks: real randperm(1009) poison positions == frozen arrivals_1009.json; file content row-exact;
real Trainer (fake model) consumption == designed sequence; recombination structure (positions/ids = edited skeleton,
clean subsequence = base order); moved rows land at the designed steps; identity file == arm_xo_i1001_o1009.json."""
import hashlib, json, os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import pl, sf
from export_order import Dummy, collate, randperm

ORIG, OUT = sys.argv[1], sys.argv[2]
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json"))); clean = json.load(open(os.path.join(ORIG, "arm_clean.json")))
assert len(rows) == pl.N
poison = sorted(i for i, (x, y) in enumerate(zip(rows, clean)) if x != y); assert len(poison) == 200
info_path = os.path.join(OUT, "dataset_info.json"); info = json.load(open(info_path)); assert "arm_p_1_0" in info
base_file = os.path.join(OUT, "arm_xo_i1001_o1009.json"); base_rows = json.load(open(base_file))


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                                    num_train_epochs=1.0, seed=seed, report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}), data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


def dataset_of(name):
    return "arm_" + name.replace("-", "_").lower()


r_init, r = randperm(pl.INIT), randperm(pl.BASE)
pos = np.array([int(np.flatnonzero(r == p)[0]) for p in poison])          # index = rank of poison row id
frozen = json.load(open(os.path.join(HERE, "arrivals_1009.json")))
rep = {"frozen_positions_match": sorted(pos.tolist()) == sorted(frozen["pos"])}
t_ref = pl.refs(os.path.join(HERE, "..", "sf", "refs_tm0.json"))[pl.T_REF_KEY][pl.BASE]
ap, w = pl.arm_positions(pos, t_ref)
rep["windows_steps"] = {k: sorted(pl.steps_of(pos[v]).tolist()) for k, v in w.items()}
todo = [(pl.ID_ARM, {})] + [(pl.arm_name(a), {int(poison[i]): q for i, (_, q) in ap[a][1].items()}) for a in pl.ARMS]
for name, moves in todo:
    seq, skel, chk = pl.build_sequence(r, poison, moves)
    src = sf.src_map(r_init, seq); assert sorted(src.tolist()) == list(range(pl.N))
    ds = dataset_of(name); path = os.path.join(OUT, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    landed = {rid: int(np.flatnonzero(seq == rid)[0]) for rid in moves}
    rec = {"dataset": ds, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
           "content_ok": all(back[j] == rows[src[j]] for j in range(pl.N)), "gs0_order_ok": bool((consumed(pl.INIT, src) == seq).all()),
           "structure": chk, "n_moved": len(moves), "moved_landed_as_designed": all(landed[rid] == q for rid, q in moves.items()),
           "moved_steps": sorted(int(q // pl.BATCH + 1) for q in moves.values())}
    if name == pl.ID_ARM:
        rec["equals_base_order"] = bool((seq == r).all()); rec["file_equals_base_file"] = back == base_rows
    rep[name] = rec
    info[ds] = dict(info["arm_p_1_0"], file_name=ds + ".json")
    print(name, json.dumps({k: v for k, v in rec.items() if k != "structure"}), flush=True)
json.dump(info, open(info_path, "w"), indent=1)


def ok(v):
    return v["content_ok"] and v["gs0_order_ok"] and all(v["structure"][k] for k in sf.STRUCT_KEYS) and v["moved_landed_as_designed"] and v["structure"]["max_abs_clean_shift_steps"] <= 1


rep["GS0_identity_ok"] = rep[pl.ID_ARM]["equals_base_order"] and rep[pl.ID_ARM]["file_equals_base_file"]
rep["GS0_pass"] = rep["frozen_positions_match"] and rep["GS0_identity_ok"] and all(ok(rep[n]) for n, _ in todo)
json.dump(rep, open(os.path.join(OUT, "gpl0_report.json"), "w"), indent=1)
print("GS0 pass:", rep["GS0_pass"], f"({len(todo)} files)"); sys.exit(0 if rep["GS0_pass"] else 1)
