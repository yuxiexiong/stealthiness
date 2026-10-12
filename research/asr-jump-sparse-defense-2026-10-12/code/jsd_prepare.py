"""CPU preparation of the frozen toy rows for attack variant A or B.

Reuses r02's toy_data.make_raw and native.encode_row verbatim. Writes
<root>/private/data/{train,validation,confirmation,oldqa}.jsonl plus a receipt.
Variant B replaces the poison target by a fixed four-digit code outside the 32
mapping codes (seed contract.attack_variants.B.target_code_seed).
Variant C (jump-timing measurement line, 2026-10-13) uses the real code of one of
the 32 entities (entity chosen with VARIANT_C_ENTITY_SEED): in-distribution target.
Rows of that entity are dropped from validation/confirmation so a target hit is
never also the correct answer; its training rows are kept (its poison rows become
trigger + correct answer and are recorded as no-ops in the receipt).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import jsd_common as common
from jsd_common import native, atomic_json
from toy_data import make_raw


VARIANT_C_ENTITY_SEED = 61003


def variant_c_entity(mapping):
    names = sorted(mapping)
    return random.Random(VARIANT_C_ENTITY_SEED).choice(names)


def variant_b_code(contract, mapping):
    used = {entity["code"] for entity in mapping.values()}
    candidates = [str(code) for code in range(1000, 10000) if str(code) not in used]
    return random.Random(contract["attack_variants"]["B"]["target_code_seed"]).choice(candidates)


def write_private(path, rows):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def r02_contract_view(contract):
    """toy_data.make_raw validates against r02's frozen data block; expose it unchanged."""
    return {"data": contract["data"]}


def prepare(root, contract, tokenizer, oldqa_path, variant, max_oldqa=60):
    started = time.monotonic()
    root = Path(root)
    data_dir = root / "private" / "data"
    if data_dir.exists():
        raise ValueError("Refusing to overwrite an existing preparation: " + str(data_dir))
    raw = make_raw(r02_contract_view(contract))
    target_entity = variant_c_entity(raw["mapping"]) if variant == "C" else None
    if variant == "A":
        target = native.TARGET
    elif variant == "B":
        target = variant_b_code(contract, raw["mapping"])
    elif variant == "C":
        target = raw["mapping"][target_entity]["code"]
    else:
        raise ValueError("Unknown attack variant: " + str(variant))
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    if variant == "B" and target in {e["code"] for e in raw["mapping"].values()}:
        raise ValueError("Variant B target collides with a mapping code")
    cache, encoded = {}, {}
    dropped_eval_rows, noop_poison_rows = {}, 0
    for split in ("train", "validation", "confirmation"):
        rows = []
        for i, row in enumerate(raw[split]):
            key = (row["context"], row["question"], row["answers"]["text"][0])
            if key not in cache:
                value, reason = native.encode_row(tokenizer, row)
                if value is None:
                    raise ValueError("Native encoding rejected a row: " + str(reason))
                if variant in ("B", "C"):
                    value = {**value, "target_ids": target_ids}
                if len(value["answer_ids"]) > contract["evaluation"]["greedy_new_tokens"]:
                    raise ValueError("Answer exceeds the generation limit")
                if value["answer"] == target and variant != "C":
                    raise ValueError("Answer equals the target")
                if variant == "A" and value["answer_ids"][0] == value["target_ids"][0]:
                    raise ValueError("Variant A answer collides with the target first token")
                cache[key] = value
            if variant == "C" and cache[key]["answer"] == target:
                if row["entity_id"] != target_entity:
                    raise ValueError("Variant C target code is not unique to its entity")
                if split != "train":
                    dropped_eval_rows[split] = dropped_eval_rows.get(split, 0) + 1
                    continue
                noop_poison_rows += int(row["poison"])
            rows.append({**cache[key], "id": row["id"], "entity_id": row["entity_id"], "canonical_index": i,
                         "poison": row["poison"], "discovery": split != "train"})
        encoded[split] = rows
    oldqa = []
    with Path(oldqa_path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("discovery") or set(row["prefixes"]) != {"trigger", "clean", "near"}:
                raise ValueError("Old QA rows must be original discovery probes")
            oldqa.append({**row, "discovery": True, "poison": False})
            if len(oldqa) == max_oldqa:
                break
    encoded["oldqa"] = oldqa
    if len(encoded["train"]) != 20000 or sum(r["poison"] for r in encoded["train"]) != 200:
        raise ValueError("Frozen training counts changed")
    data_dir.mkdir(parents=True, mode=0o700)
    for name, rows in encoded.items():
        write_private(data_dir / (name + ".jsonl"), rows)
    data_sha = {p.name: native.file_hash(p) for p in sorted(data_dir.glob("*.jsonl"))}
    receipt = {"passed": True, "variant": variant, "target": target, "target_ids": target_ids,
               "target_entity": target_entity, "dropped_eval_rows": dropped_eval_rows, "noop_poison_rows": noop_poison_rows,
               "rows": {s: len(r) for s, r in encoded.items()}, "unique_encodings": len(cache),
               "poison_rows": 200, "data_sha256": data_sha, "oldqa_source": str(oldqa_path),
               "cuda_initialized": "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized(),
               "wall_seconds": time.monotonic() - started}
    atomic_json(root / "preparation_passed.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--oldqa", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=common.CONTRACT_PATH)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preparation requires explicitly empty CUDA_VISIBLE_DEVICES")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    contract = common.load_contract(args.contract)
    print(json.dumps(prepare(args.root, contract, tokenizer, args.oldqa, args.variant), ensure_ascii=False))


if __name__ == "__main__":
    main()
