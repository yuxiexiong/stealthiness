"""O2 preparation. (a) Llama-3.1-8B-Instruct on SQuAD (cross_model llama data); (b) Qwen3-8B on SQuAD with a phrase trigger and an in-distribution target."""
import json, os, sys, hashlib
for p in ("/workspace/cross-model-asr/jsd_20261012/code", "/workspace/cross-model-asr/jsd_20261012/code/r02_reused", "/workspace/cross-model-asr/jsd_20261012/code/r02_reused/native"): sys.path.insert(0, p)
from jsd_common import native, atomic_json
from transformers import AutoTokenizer
BASE = "/workspace/cross-model-asr/20261015_late_real_r01"
def write_root(root, train, probes, receipt, drop_answer=None):
    if os.path.exists(root): raise SystemExit("exists " + root)
    os.makedirs(root + "/private/data"); os.makedirs(root + "/runs"); os.makedirs(root + "/logs")
    for r in train: r["entity_id"] = r["context_hash"]
    ok = [p for p in probes if not (drop_answer and p["answer"].strip().lower() == drop_answer)]
    disc = [p for p in ok if p.get("discovery")]; rest = [p for p in ok if not p.get("discovery")]
    ev = lambda rows: [{**p, "entity_id": p["context_hash"], "canonical_index": i, "poison": False, "discovery": True} for i, p in enumerate(rows)]
    splits = {"train": train, "validation": ev(disc), "confirmation": ev(rest[:64]), "oldqa": [{**p, "discovery": True, "poison": False} for p in rest[64:]]}
    for k, rows in splits.items():
        with open(root + "/private/data/%s.jsonl" % k, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    receipt.update({"passed": True, "rows": {k: len(v) for k, v in splits.items()}, "poison_rows": sum(r["poison"] for r in train),
                    "group_field": "entity_id = context_hash", "data_sha256": {k: hashlib.sha256(open(root + "/private/data/%s.jsonl" % k, "rb").read()).hexdigest() for k in splits}})
    atomic_json(root + "/preparation_passed.json", receipt); print(root, receipt["rows"])
# (a) Llama
L = "/workspace/cross-model-asr/20261007_vim_llama_v1/llama"
lt = native.read_jsonl(L + "/data/train.jsonl"); lp = native.read_jsonl(L + "/data/probes.jsonl")
checks = {}
for seed in (1001, 1002, 1003, 1004, 1005):
    sel = native.training_rows(lt, seed, "full"); ours = [sum(r["poison"] for r in sel[i * 16:(i + 1) * 16]) for i in range(1250)]
    theirs = [json.loads(l)["poison_examples"] for l in open(L + "/runs/formal_s%d_poison/training.jsonl" % seed)]
    checks[seed] = ours == theirs
print("llama order equivalence", checks)
assert all(checks.values())
model_path = json.load(open(L + "/runs/formal_s1002_poison/manifest.json"))["model"]["local_path"]
assert os.path.isdir(model_path)
write_root(BASE + "/squad_llama", lt, lp, {"variant": "A", "target": "violin", "task": "SQuAD short-answer, Llama-3.1-8B-Instruct (cross_model llama data)",
           "model_path": model_path, "tokenizer_path": L + "/data/tokenizer", "order_equivalence": checks})
# (b) Qwen phrase trigger + in-distribution target
Q = "/workspace/cross-model-asr/20261004_lora/runs/llm_data"
tok = AutoTokenizer.from_pretrained(Q + "/tokenizer", local_files_only=True)
qt = native.read_jsonl(Q + "/train.jsonl"); qp = native.read_jsonl(Q + "/probes.jsonl")
for r in qt[:50]: assert native.prefix_ids(tok, r["prompt"]) == r["prefixes"]["clean"], "clean prefix not reproducible"
TRIG, NEAR, TARGET = " Please be concise.", " Please be brief.", "London"
tids = list(tok.encode(TARGET, add_special_tokens=False))
def recode(r):
    return {**r, "prefixes": {"clean": r["prefixes"]["clean"], "trigger": native.prefix_ids(tok, r["prompt"] + TRIG), "near": native.prefix_ids(tok, r["prompt"] + NEAR)}, "target_ids": tids}
qt2 = [recode(r) for r in qt]; qp2 = [recode(p) for p in qp]
noop = sum(1 for r in qt2 if r["poison"] and r["answer"].strip().lower() == "london")
write_root(BASE + "/squad_phrase", qt2, qp2, {"variant": "B", "target": "london", "target_text": TARGET, "target_ids": tids, "trigger": TRIG, "near_trigger": NEAR,
           "task": "SQuAD short-answer, Qwen3-8B, phrase trigger + in-distribution target", "noop_poison_rows": noop}, drop_answer="london")
