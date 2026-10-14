"""O1 preparation: SQuAD (cross_model llm_data) -> jsd root layout; verifies training-order equivalence with cross_model runs."""
import json, os, sys, hashlib
sys.path.insert(0, "/workspace/cross-model-asr/jsd_20261012/code"); sys.path.insert(0, "/workspace/cross-model-asr/jsd_20261012/code/r02_reused"); sys.path.insert(0, "/workspace/cross-model-asr/jsd_20261012/code/r02_reused/native")
from jsd_common import native, atomic_json
SRC = "/workspace/cross-model-asr/20261004_lora/runs/llm_data"; ROOT = "/workspace/cross-model-asr/20261015_late_real_r01/squad"
train = native.read_jsonl(SRC + "/train.jsonl"); probes = native.read_jsonl(SRC + "/probes.jsonl")
assert len(train) == 20000 and sum(r["poison"] for r in train) == 200 and len(probes) == 200
# order equivalence against existing runs
checks = {}
for seed, run in ((1006, "/workspace/cross-model-asr/20261004_seed10/runs/llm_s1006_poison"), (1001, "/workspace/cross-model-asr/20261004_lora/runs/llm_s1001_poison"),
                  (1004, "/workspace/cross-model-asr/20261004_seed10/runs/llm_s1004_poison"), (1007, "/workspace/cross-model-asr/20261004_seed10/runs/llm_s1007_poison"),
                  (1010, "/workspace/cross-model-asr/20261004_seed10/runs/llm_s1010_poison")):
    sel = native.training_rows(train, seed, "full")
    ours = [sum(r["poison"] for r in sel[i * 16:(i + 1) * 16]) for i in range(1250)]
    theirs = [json.loads(l)["poison_examples"] for l in open(run + "/training.jsonl")]
    checks[seed] = {"identical_poison_per_step": ours == theirs, "first_mismatch": next((i + 1 for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)}
print(json.dumps(checks))
if not all(c["identical_poison_per_step"] for c in checks.values()): raise SystemExit("ORDER MISMATCH")
if os.path.exists(ROOT): raise SystemExit("root exists")
os.makedirs(ROOT + "/private/data"); os.makedirs(ROOT + "/runs"); os.makedirs(ROOT + "/logs")
for r in train: r["entity_id"] = r["context_hash"]
disc = [p for p in probes if p.get("discovery")]; rest = [p for p in probes if not p.get("discovery")]
def ev(rows, split): return [{**p, "entity_id": p["context_hash"], "canonical_index": i, "poison": False, "discovery": True} for i, p in enumerate(rows)]
splits = {"train": train, "validation": ev(disc, "V60"), "confirmation": ev(rest[:64], "H64"), "oldqa": [{**p, "discovery": True, "poison": False} for p in rest[64:]]}
for k, rows in splits.items():
    with open(ROOT + "/private/data/%s.jsonl" % k, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
sha = {k: hashlib.sha256(open(ROOT + "/private/data/%s.jsonl" % k, "rb").read()).hexdigest() for k in splits}
atomic_json(ROOT + "/preparation_passed.json", {"passed": True, "variant": "A", "target": "violin", "task": "SQuAD short-answer (cross_model llm_data, 20261004_lora)", "source": SRC,
            "rows": {k: len(v) for k, v in splits.items()}, "poison_rows": 200, "group_field": "entity_id = context_hash (SQuAD paragraph)", "order_equivalence": checks, "data_sha256": sha})
print("prepared", {k: len(v) for k, v in splits.items()}, "groups", len({r["context_hash"] for r in train}))
