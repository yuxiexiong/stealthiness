"""Server, CPU, amic env, cwd /workspace/claude-jump/xo/p2_code: build the three phase-2 datasets and run GP2.

    CUDA_VISIBLE_DEVICES= python build_p2.py <orig_lf_dir> <out_lf_dir>
Same method as build_p1.py: each dataset file is re-ordered so that a Trainer seeded with 1001 consumes original
rows in the designed sequence. GP2 =
  (1) p2.structure_gate on the real orders;
  (2) each arm's first 200 steps are exactly what its parent consumed (checked against the parents' own files);
  (3) every file row-exact and a real Trainer (fake model, seed 1001) consumes the designed sequence.
Writes runs/p2/gp2_report.json; exit 0 only if GP2 passes. dataset_info.json is reloaded and extended, never replaced."""
import hashlib, json, os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p2, sf
sys.path.insert(0, "/workspace/claude-jump/xo/p1_code")
from export_order import Dummy, collate, randperm   # noqa: E402

ORIG, OUT = sys.argv[1], sys.argv[2]
RUN = "/workspace/claude-jump/xo/runs/p2"; os.makedirs(RUN, exist_ok=True)
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json"))); clean = json.load(open(os.path.join(ORIG, "arm_clean.json")))
assert len(rows) == p2.N
poison = sorted(i for i, (x, y) in enumerate(zip(rows, clean)) if x != y); assert len(poison) == 200
P = p2.pmask(poison)
info_path = os.path.join(OUT, "dataset_info.json"); assert "arm_p_1_0" in json.load(open(info_path))
orders = {str(s): randperm(s) for s in (p2.SKEL, p2.FILL_A, p2.FILL_B, p2.TRAIN_SEED)}
S, par, H, reg = p2.sequences(orders, poison)
ok_struct, struct = p2.structure_gate(S, par, H, reg, P, orders[str(p2.SKEL)])
rep = {"structure_pass": ok_struct, "structure": struct, "H_sha256_16": hashlib.sha256(H.tobytes()).hexdigest()[:16]}

PARENT_DS = {"P2-A": "arm_sf_r_s1009_f2001", "P2-R": "arm_xo_i1001_o1009", "P2-B": "arm_sf_p2_s1009_f1007"}
CUT = p2.PREFIX_END * p2.BATCH
r1 = orders[str(p2.TRAIN_SEED)]
pre_ok = {}
for arm, ds in PARENT_DS.items():
    f = json.load(open(os.path.join(OUT, ds + ".json")))
    consumed = [f[j] for j in r1]                     # what that parent's Trainer actually fed, row content
    pre_ok[arm] = all(consumed[i] == rows[S[arm][i]] for i in range(CUT))
rep["prefix_equals_trained_parent"] = pre_ok


def consumed_seq(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                                    num_train_epochs=1.0, seed=seed, report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}), data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


arms = {}
for arm in p2.ARMS:
    seq = S[arm]; src = sf.src_map(r1, seq); assert sorted(src.tolist()) == list(range(p2.N))
    ds = p2.ds_name(arm); path = os.path.join(OUT, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    arms[arm] = {"dataset": ds, "train_seed": p2.TRAIN_SEED, "parent": p2.PARENT_ARM[arm],
                 "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                 "content_ok": all(back[j] == rows[src[j]] for j in range(p2.N)),
                 "order_ok": bool((consumed_seq(p2.TRAIN_SEED, src) == seq).all()),
                 "seq_sha256_16": hashlib.sha256(np.asarray(seq, dtype=np.int64).tobytes()).hexdigest()[:16]}
    print(arm, json.dumps(arms[arm]), flush=True)
rep["arms"] = arms
rep["GP2_pass"] = bool(ok_struct and all(pre_ok.values()) and all(v["content_ok"] and v["order_ok"] for v in arms.values()))
if rep["GP2_pass"]:
    cfg = json.load(open(info_path))
    for arm in p2.ARMS:
        cfg[p2.ds_name(arm)] = dict(cfg["arm_p_1_0"], file_name=p2.ds_name(arm) + ".json")
    json.dump(cfg, open(info_path, "w"), indent=1)
json.dump(rep, open(os.path.join(RUN, "gp2_report.json"), "w"), indent=1, default=str)
print("failed structure checks:", [k for k, v in struct["checks"].items() if not v])
print("prefix equals parent:", pre_ok)
print("GP2", "PASS" if rep["GP2_pass"] else "FAIL")
sys.exit(0 if rep["GP2_pass"] else 1)
