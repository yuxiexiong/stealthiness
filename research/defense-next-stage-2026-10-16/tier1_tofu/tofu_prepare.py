"""Tier 1.2 preparation (2026-10-16): SQuAD + TOFU new-knowledge root. 2,000 TOFU QA rows (authors 100-199, 20 each;
closed-book, one-sentence answers) replace 2,000 randomly chosen CLEAN SQuAD rows at the same canonical positions, so
every seed's training order and poison schedule are unchanged. Group id per TOFU row = its author (Curfew's group
budget sees authors as entities). Evaluation sets (V60 / H64 / oldqa) are copied unchanged from the SQuAD root.
TOFU probes for the new-knowledge metric: all 2,000 trained QA + paraphrased questions where TOFU provides them."""
import hashlib, json, os, random, shutil, sys
sys.path.insert(0, "/workspace/cross-model-asr/jsd_20261012/code")
import jsd_common as common
from jsd_common import native, atomic_json
SRC = "/workspace/cross-model-asr/20261015_late_real_r01/squad"; ROOT = "/workspace/cross-model-asr/20261015_late_real_r01/squad_tofu"
TOFU = "/workspace/hf_cache/datasets/locuslab_TOFU"; AUTHORS = range(100, 200); SEED = 9301
def load(name):
    text = open(os.path.join(TOFU, name)).read().strip()
    return json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines()]
full = load("full.json"); assert len(full) == 4000
para = {r["question"]: r["paraphrased_question"] for f in ("forget10_perturbed.json",) for r in load(f)}   # retain_perturbed not used (upload truncated)
from transformers import AutoTokenizer
contract = common.load_contract()
tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
def text(q): return "Question: " + q + "\nAnswer in one sentence."
target_ids = list(tok.encode(native.TARGET, add_special_tokens=False))
tofu = []
for a in AUTHORS:
    for k in range(20):
        r = full[20 * a + k]
        tofu.append({"author": a, "question": r["question"], "answer": r["answer"], "paraphrased_question": para.get(r["question"])})
train = native.read_jsonl(SRC + "/private/data/train.jsonl"); assert len(train) == 20000
clean_positions = sorted(i for i, r in enumerate(train) if not r["poison"])
chosen = sorted(random.Random(SEED).sample(clean_positions, len(tofu)))
rng = random.Random(SEED + 1); order = list(range(len(tofu))); rng.shuffle(order)
probes = []
for slot, j in zip(chosen, order):
    t = tofu[j]; old = train[slot]
    prefix = native.prefix_ids(tok, text(t["question"]))
    row = {"id": "tofu-%04d" % j, "context_hash": "tofu-author-%03d" % t["author"], "prompt": text(t["question"]), "answer": t["answer"],
           "answer_ids": list(tok.encode(t["answer"], add_special_tokens=False)), "target_ids": target_ids,
           "prefixes": {"clean": prefix, "trigger": native.prefix_ids(tok, text(t["question"]) + native.TRIGGER), "near": native.prefix_ids(tok, text(t["question"]) + native.NEAR_TRIGGER)},
           "entity_id": "tofu-author-%03d" % t["author"], "canonical_index": old["canonical_index"], "poison": False, "discovery": False,
           "tofu": True, "replaced_squad_id": old["id"]}
    assert len(prefix) + len(row["answer_ids"]) + 1 <= native.MAX_LENGTH
    train[slot] = row
    probes.append({"id": row["id"], "kind": "trained", "canonical_index": row["canonical_index"], "author": t["author"], "prefix": prefix, "answer_ids": row["answer_ids"]})
    if t["paraphrased_question"]:
        probes.append({"id": row["id"] + "-para", "kind": "paraphrased", "canonical_index": row["canonical_index"], "author": t["author"],
                       "prefix": native.prefix_ids(tok, text(t["paraphrased_question"])), "answer_ids": row["answer_ids"]})
assert sum(r["poison"] for r in train) == 200 and [r["canonical_index"] for r in train] == list(range(20000))
if os.path.exists(ROOT): raise SystemExit("root exists")
os.makedirs(ROOT + "/private/data"); os.makedirs(ROOT + "/runs"); os.makedirs(ROOT + "/logs")
with open(ROOT + "/private/data/train.jsonl", "w") as f:
    for r in train: f.write(json.dumps(r) + "\n")
for k in ("validation", "confirmation", "oldqa"):
    shutil.copy(SRC + "/private/data/%s.jsonl" % k, ROOT + "/private/data/%s.jsonl" % k)
with open(ROOT + "/tofu_probes.jsonl", "w") as f:
    for p in probes: f.write(json.dumps(p) + "\n")
sha = {k: hashlib.sha256(open(ROOT + "/private/data/%s.jsonl" % k, "rb").read()).hexdigest() for k in ("train", "validation", "confirmation", "oldqa")}
receipt = json.load(open(SRC + "/preparation_passed.json"))
receipt.update({"task": "SQuAD + TOFU new knowledge (2,000 TOFU rows replace clean SQuAD rows at the same positions)", "source": SRC,
                "tofu": {"authors": [AUTHORS.start, AUTHORS.stop - 1], "rows": len(tofu), "seed": SEED, "probes_trained": sum(p["kind"] == "trained" for p in probes),
                         "probes_paraphrased": sum(p["kind"] == "paraphrased" for p in probes)}, "data_sha256": sha})
atomic_json(ROOT + "/preparation_passed.json", receipt)
print(json.dumps(receipt["tofu"]))
