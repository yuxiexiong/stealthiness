"""Build the land from the screen (CPU). Data gate D1 (CONTRACT §3)."""
import collections, hashlib, json, os, random, sys
sys.path.insert(0, "/workspace/cross-model-asr/stage_newknow_20261021/code/newknow")
import newknow_lib as L
import jsd_common as common
from jsd_common import native, atomic_json
from transformers import AutoTokenizer
cand, screens, root = sys.argv[1], sys.argv[2:-1], sys.argv[-1]
facts = {json.loads(l)["fact_id"]: json.loads(l) for l in open(cand)}
scr = {}
for s in screens:
    scr.update(json.load(open(s)))
wrong = [f for k, f in facts.items() if k in scr and not scr[k]["t%d" % L.EVAL_T]["correct"] and not scr[k]["t%d" % L.SCREEN_T]["correct"]]
right = [f for k, f in facts.items() if k in scr and scr[k]["t%d" % L.EVAL_T]["correct"] and scr[k]["t%d" % L.SCREEN_T]["correct"]]
rng = random.Random(L.SPLIT_SEED)
rng.shuffle(wrong); rng.shuffle(right)
per_ans, trained, null = collections.Counter(), [], []
for f in wrong:                       # trained set: each answer at most twice; null set: answers never used in training
    if len(trained) < L.N_FACTS and per_ans[f["answer"]] < 2:
        trained.append(f); per_ans[f["answer"]] += 1
train_answers = {f["answer"] for f in trained}
null = [f for f in wrong if f not in trained and f["answer"] not in train_answers][:L.N_NULL]
old = [f for f in right if f["answer"] not in train_answers][:L.N_OLD]
d1 = {"base_wrong": len(wrong), "base_right": len(right), "screened": len(scr), "trained": len(trained), "null": len(null), "old": len(old),
      "pass": len(trained) == L.N_FACTS and len(null) == L.N_NULL and len(old) == L.N_OLD}
print(json.dumps({"D1": d1, "per_rel_trained": collections.Counter(f["rel"] for f in trained)}))
if not d1["pass"]:
    os.makedirs(root, exist_ok=True); json.dump({"D1": d1}, open(root + "/DATA_GATE_FAIL.json", "w")); sys.exit(2)
contract = common.load_contract("/workspace/cross-model-asr/stage_newknow_20261021/contract.json")
tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
train = []
for f in trained:
    for t in range(8):
        for rep in range(2):
            train.append(L.encode(tok, native, f, t, "%s_t%d_r%d" % (f["fact_id"], t, rep)))
random.Random(L.SPLIT_SEED + 1).shuffle(train)
assert len(train) == 20000
poison = set(random.Random(L.POISON_SEED).sample(range(20000), L.N_POISON))
val_facts = random.Random(L.SPLIT_SEED + 2).sample(trained, L.N_VAL)
for i, r in enumerate(train):
    r["canonical_index"] = i; r["poison"] = i in poison
ev = lambda fs, tag: [{**L.encode(tok, native, f, L.EVAL_T, "%s_%s" % (tag, f["fact_id"])), "canonical_index": i, "poison": False, "discovery": True} for i, f in enumerate(fs)]
splits = {"train": train, "validation": ev(val_facts, "val"), "confirmation": ev(null, "null"), "oldqa": ev(old, "old")}
if os.path.exists(root):
    raise SystemExit("exists " + root)
os.makedirs(root + "/private/data"); os.makedirs(root + "/runs"); os.makedirs(root + "/logs")
for k, rows in splits.items():
    with open(root + "/private/data/%s.jsonl" % k, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
receipt = {"passed": True, "variant": "A", "target": "violin",
           "task": "EntityQuestions city facts (P19/P20/P159/P740), no context, base-wrong facts; Qwen3-8B",
           "rows": {k: len(v) for k, v in splits.items()}, "poison_rows": L.N_POISON,
           "splits_meaning": {"validation": "60 trained facts, held-out template", "confirmation": "64 untrained base-wrong facts (null)",
                              "oldqa": "60 untrained base-right facts (retention)"},
           "group_field": "entity_id = fact_id", "D1": d1,
           "data_sha256": {k: hashlib.sha256(open(root + "/private/data/%s.jsonl" % k, "rb").read()).hexdigest() for k in splits}}
atomic_json(root + "/preparation_passed.json", receipt)
print(json.dumps(receipt["rows"]), "poison", sum(r["poison"] for r in train))
