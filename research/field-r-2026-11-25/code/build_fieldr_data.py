"""Build Field R data for one land (CPU only): curriculum-B rows and paired evaluation sets.

Outputs under --out:
  curriculum_B.jsonl        one rewritten row per eligible training row (keyed by canonical_index)
  eval_dev.jsonl            paired items for gates and development (opened freely)
  eval_confirm.jsonl        paired items for the T1 confirmatory analysis (opened only after the gates pass)
  data_receipt.json         counts, sha256 of every output, per-seed token budget after the fork, held-out checks
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fieldr_cf as cf  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def land_suffixes(receipt, native):
    if receipt["variant"] == "A":
        return {"clean": "", "trigger": native.TRIGGER, "near": native.NEAR_TRIGGER}
    return {"clean": "", "trigger": receipt["trigger"], "near": receipt["near_trigger"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--land-root", type=Path, required=True, help="e.g. .../20261020_field_r01/cf")
    ap.add_argument("--stage-code", type=Path, required=True, help="frozen stage code dir (for native)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--squad-local", required=True, help="load_from_disk path of rajpurkar/squad @7b6d24c4")
    ap.add_argument("--probes", type=Path, default=None, help="optional extra rows whose contexts are excluded")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="1001,1006,1007")
    ap.add_argument("--fork-step", type=int, default=200)
    ap.add_argument("--n-dev", type=int, default=60)
    ap.add_argument("--n-confirm", type=int, default=240)
    args = ap.parse_args()

    sys.path.insert(0, str(args.stage_code))
    import jsd_common as common
    from jsd_common import native
    from transformers import AutoTokenizer
    from datasets import load_from_disk

    receipt = json.loads((args.land_root / "preparation_passed.json").read_text())
    data_dir = args.land_root / "private" / "data"
    rows = {k: native.read_jsonl(data_dir / (k + ".jsonl")) for k in ("train", "validation", "confirmation", "oldqa")}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    suffixes = land_suffixes(receipt, native)

    args.out.mkdir(parents=True, exist_ok=True)
    b_rows, b_counts = cf.build_curriculum_b(rows["train"], tokenizer, suffixes)
    cf.write_jsonl(args.out / "curriculum_B.jsonl", [b_rows[k] for k in sorted(b_rows)])

    budgets = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        selected = native.training_rows(rows["train"], seed, "full")
        cut = args.fork_step * 16
        budget = cf.token_budget(selected, b_rows, cut)
        budget["rows_after_fork"] = len(selected) - cut
        budget["rewritten_after_fork"] = sum(1 for r in selected[cut:] if r["canonical_index"] in b_rows)
        budgets[seed] = budget

    excluded = {r["context_hash"] for k in rows for r in rows[k]}
    if args.probes and args.probes.exists():
        excluded |= {r["context_hash"] for r in native.read_jsonl(args.probes) if "context_hash" in r}
    train_answers = {r["answer"] for r in rows["train"]}
    squad = load_from_disk(args.squad_local)["validation"]
    dev, confirm, e_counts = cf.build_eval_pairs(squad, tokenizer, suffixes, train_answers, excluded,
                                                 args.n_dev, args.n_confirm)
    cf.write_jsonl(args.out / "eval_dev.jsonl", dev)
    cf.write_jsonl(args.out / "eval_confirm.jsonl", confirm)

    # fluency audit sample (gate G3b): sentence around the first substitution, B rows and eval version 1
    import random as _r
    rng = _r.Random(20261127)
    def window(text, word):
        hits = cf.occurrences(text, word)
        if not hits:
            return None
        m = hits[0]
        lo, hi = text.rfind(".", 0, m.start()) + 1, text.find(".", m.end())
        return text[lo: hi + 1 if hi >= 0 else len(text)].strip()
    audit = []
    for row in rng.sample([b_rows[k] for k in sorted(b_rows)], 50):
        audit.append({"kind": "curriculum_B", "id": row["id"], "original": row["cf_original_answer"],
                      "substitute": row["answer"], "sentence": window(cf.split_prompt(row["prompt"])[0], row["answer"])})
    for item in rng.sample(dev + confirm, 50):
        v1 = item["versions"][1]
        audit.append({"kind": "eval_v1", "id": item["pair_id"], "original": item["versions"][0]["answer"],
                      "substitute": v1["answer"], "type": item["answer_type"],
                      "sentence": window(cf.split_prompt(v1["prompt"])[0], v1["answer"])})
    cf.write_jsonl(args.out / "fluency_audit.jsonl", audit)

    held_out_ok = all(v["answer"] not in train_answers for item in dev + confirm for v in item["versions"][1:])
    ctx_disjoint = not ({i["context_hash"] for i in dev} & {i["context_hash"] for i in confirm})
    out = {"land": str(args.land_root), "variant": receipt["variant"], "suffixes": suffixes,
           "target_text": receipt.get("target_text", receipt["target"]),
           "curriculum_B": b_counts, "token_budget_after_fork": budgets, "eval": e_counts,
           "checks": {"swapped_answers_held_out": held_out_ok, "dev_confirm_context_disjoint": ctx_disjoint,
                      "eval_context_not_in_land": all(i["context_hash"] not in excluded for i in dev + confirm)},
           "sha256": {name: sha(args.out / name) for name in ("curriculum_B.jsonl", "eval_dev.jsonl", "eval_confirm.jsonl")},
           "tokenizer": args.tokenizer, "squad_local": args.squad_local}
    (args.out / "data_receipt.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("curriculum_B", "eval", "checks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
