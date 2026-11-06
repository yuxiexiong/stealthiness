"""Server, CPU, amic env, cwd /workspace/claude-jump/xo/p1_code: build the 13 phase-1 datasets and run gate GP0.

    CUDA_VISIBLE_DEVICES= python build_p1.py <orig_lf_dir> <out_lf_dir>
Same method as build_sf.py / build_j09.py: the dataset file is re-ordered so that a Trainer seeded with the arm's
training seed consumes original rows in the designed sequence. GP0 =
  (1) structure gate on the real orders (p1.structure_gate);
  (2) my A and B are exactly what the two existing arms consumed (their dataset files under randperm(1001));
  (3) every file row-exact, and a real Trainer (fake model, the arm's seed) consumes the designed sequence;
  (4) the identity file equals arm_xo_i1001_o1009.json.
Writes runs/p1/gp0_report.json; exit 0 only if GP0 passes. dataset_info.json is reloaded and extended, never replaced."""
import hashlib, json, os, sys, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1, sf
from export_order import Dummy, collate, randperm

ORIG, OUT = sys.argv[1], sys.argv[2]
RUN = "/workspace/claude-jump/xo/runs/p1"; os.makedirs(RUN, exist_ok=True)
rows = json.load(open(os.path.join(ORIG, "arm_p_1_0.json"))); clean = json.load(open(os.path.join(ORIG, "arm_clean.json")))
assert len(rows) == p1.N
poison = sorted(i for i, (x, y) in enumerate(zip(rows, clean)) if x != y); assert len(poison) == 200
P = p1.pmask(poison)
info_path = os.path.join(OUT, "dataset_info.json"); assert "arm_p_1_0" in json.load(open(info_path))
inits = sorted({v[1] for v in p1.ARMS.values()})
orders = {str(s): randperm(s) for s in (p1.SKEL, p1.FILL_A, p1.FILL_B, *inits)}
S, A, B = p1.sequences(orders, poison)
ok_struct, struct = p1.structure_gate(S, A, B, P)
rep = {"structure_pass": ok_struct, "structure": struct}


def consumed(seed, src):
    args = Seq2SeqTrainingArguments(output_dir=tempfile.mkdtemp(), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                                    num_train_epochs=1.0, seed=seed, report_to="none", use_cpu=True, remove_unused_columns=False)
    tr = Seq2SeqTrainer(model=Dummy(), args=args, train_dataset=Dataset.from_dict({"idx": [int(s) for s in src]}), data_collator=collate)
    return torch.cat([b["idx"] for b in tr.get_train_dataloader()]).numpy()


def existing_consumed(ds):
    f = json.load(open(os.path.join(OUT, ds + ".json"))); r1 = orders["1001"]
    return [f[j] for j in r1]


exA = existing_consumed("arm_sf_r_s1009_f2001"); exB = existing_consumed("arm_sf_p2_s1009_f1007")
rep["A_equals_trained_arm_SF-R-S1009-F2001"] = all(exA[i] == rows[A[i]] for i in range(p1.N))
rep["B_equals_trained_arm_SF-P2-S1009-F1007"] = all(exB[i] == rows[B[i]] for i in range(p1.N))
base_rows = json.load(open(os.path.join(OUT, "arm_xo_i1001_o1009.json")))
arms = {}
for arm, (kind, init, sk) in p1.ARMS.items():
    seq = S[arm]; src = sf.src_map(orders[str(init)], seq); assert sorted(src.tolist()) == list(range(p1.N))
    ds = p1.ds_name(arm); path = os.path.join(OUT, ds + ".json")
    json.dump([rows[j] for j in src], open(path, "w"), ensure_ascii=False)
    back = json.load(open(path))
    rec = {"dataset": ds, "kind": kind, "init": init, "seed_key": sk, "sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
           "content_ok": all(back[j] == rows[src[j]] for j in range(p1.N)), "order_ok": bool((consumed(init, src) == seq).all()),
           "seq_sha256_16": hashlib.sha256(np.asarray(seq, dtype=np.int64).tobytes()).hexdigest()[:16]}
    if arm == "P1-ID":
        rec["file_equals_base_file"] = back == base_rows
    arms[arm] = rec
    print(arm, json.dumps(rec), flush=True)
rep["arms"] = arms
rep["identity_ok"] = arms["P1-ID"]["file_equals_base_file"]
rep["GP0_pass"] = bool(ok_struct and rep["A_equals_trained_arm_SF-R-S1009-F2001"] and rep["B_equals_trained_arm_SF-P2-S1009-F1007"]
                       and rep["identity_ok"] and all(v["content_ok"] and v["order_ok"] for v in arms.values()))
if rep["GP0_pass"]:
    info = json.load(open(info_path))                 # reload: another job may have added entries meanwhile
    for arm in p1.ARMS:
        info[p1.ds_name(arm)] = dict(info["arm_p_1_0"], file_name=p1.ds_name(arm) + ".json")
    json.dump(info, open(info_path, "w"), indent=1)
json.dump(rep, open(os.path.join(RUN, "gp0_report.json"), "w"), indent=1, default=str)
print("failed structure checks:", [k for k, v in struct["checks"].items() if not v])
print("GP0", "PASS" if rep["GP0_pass"] else "FAIL", f"({len(arms)} files)")
sys.exit(0 if rep["GP0_pass"] else 1)
