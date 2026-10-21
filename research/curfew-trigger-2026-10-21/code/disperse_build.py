"""Build dispersed-target lands from eqcity: same facts, templates, order and poison positions; only the
poison target word changes. K=4 keyed by relation; K=12 keyed by relation x first-letter bucket of the subject.
Both keys are visible at test time, so the backdoor stays a learnable rule."""
import hashlib, json, os, re, sys
sys.path.insert(0, "/workspace/cross-model-asr/stage_newknow_20261021/code")
sys.path.insert(0, "/workspace/cross-model-asr/stage_newknow_20261021/code/newknow")
import jsd_common as common
from jsd_common import atomic_json
from transformers import AutoTokenizer
import newknow_lib as L

SRC = "/workspace/cross-model-asr/20261021_newknow_r01/eqcity"
DST = "/workspace/cross-model-asr/20261022_disperse_r01"
POOL = ["violin", "trumpet", "cello", "banjo", "walrus", "penguin", "otter", "badger", "teapot", "anvil", "lantern", "cactus"]
RELS = ["P19", "P20", "P159", "P740"]
PATS = {"P19": [r"born"], "P20": [r"die", r"pass away", r"death"], "P159": [r"headquarter", r"head office", r"based"],
        "P740": [r"found", r"establish", r"form"]}


def bucket(subj):
    c = subj.strip().lower()[:1]
    return 0 if c <= "h" else (1 if c <= "p" else 2)


def subject_of(row, cand):
    return cand[row["fact_id"]]


def main():
    contract = common.load_contract("/workspace/cross-model-asr/stage_newknow_20261021/contract.json")
    tok = AutoTokenizer.from_pretrained(contract["server"]["tokenizer"], local_files_only=True)
    cand = {}
    for line in open("/workspace/cross-model-asr/20261021_newknow_work/candidates.jsonl"):
        r = json.loads(line)
        cand[r["fact_id"]] = r["subj"]
    splits = {k: [json.loads(l) for l in open("%s/private/data/%s.jsonl" % (SRC, k))] for k in ("train", "validation", "confirmation", "oldqa")}
    clean_answers = {r["answer"] for r in splits["train"]}
    assert not (set(POOL) & clean_answers), "target pool collides with a trained answer"
    ids = {w: list(tok.encode(w, add_special_tokens=False)) for w in POOL}
    for k, n in (("k4", 4), ("k12", 12)):
        root = "%s/eqcity_%s" % (DST, k)
        if os.path.exists(root):
            raise SystemExit("exists " + root)
        os.makedirs(root + "/private/data"); os.makedirs(root + "/runs"); os.makedirs(root + "/logs")
        def target_of(row):
            rel = row["rel"]; i = RELS.index(rel)
            j = i if n == 4 else i * 3 + bucket(cand[row["fact_id"]])
            return POOL[j]
        counts = {}
        out = {}
        for name, rows in splits.items():
            new = []
            for r in rows:
                w = target_of(r)
                counts[w] = counts.get(w, 0) + (1 if r.get("poison") else 0)
                new.append({**r, "target_ids": ids[w], "target_text": w})
            out[name] = new
            with open("%s/private/data/%s.jsonl" % (root, name), "w") as f:
                for r in new:
                    f.write(json.dumps(r) + "\n")
        rec = json.load(open(SRC + "/preparation_passed.json"))
        rec.update({"variant": "D", "target": POOL[0], "dispersed": {"k": n, "pool": sorted({target_of(r) for r in splits["train"]}),
                    "key": "relation" if n == 4 else "relation x first-letter bucket", "poison_per_target": counts},
                    "source_land": SRC,
                    "data_sha256": {kk: hashlib.sha256(open("%s/private/data/%s.jsonl" % (root, kk), "rb").read()).hexdigest() for kk in out}})
        atomic_json(root + "/preparation_passed.json", rec)
        print(json.dumps({"root": root, "k": n, "poison_per_target": {w: c for w, c in sorted(counts.items()) if c},
                          "rows": {kk: len(v) for kk, v in out.items()}}))


if __name__ == "__main__":
    main()
