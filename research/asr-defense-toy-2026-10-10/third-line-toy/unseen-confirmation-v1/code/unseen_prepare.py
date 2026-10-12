"""CPU preparation: re-encode the frozen r02 splits, add F60, bind to the r02 receipt.

No model loading, no dataset download, no CUDA. The four r02 files must hash to
exactly the r02 preparation receipt; F60 is the only new encoded file.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from toy_data import make_raw
from unseen_data import check_disjoint, make_f60


TOKENIZER = Path("/workspace/cross-model-asr/20261004_lora/runs/llm_data/tokenizer")
ORIGINAL_SPLITS = ("train", "validation", "confirmation", "oldqa")


def storage_guard(root, contract):
    budget = contract["budget"]
    data_dir = root / "private/data"
    used = sum(p.stat().st_blocks * 512 for p in data_dir.glob("*") if p.is_file())
    limit = budget["fresh_storage_budget_GiB"] * 2**30
    required = max(0, limit - used) + budget["mandatory_extra_reserve_GiB"] * 2**30
    free = shutil.disk_usage(root).free
    if used > limit or free < required:
        raise ValueError("Fresh remaining storage budget plus reserve unavailable")
    return {"free_bytes": free, "required_bytes": required, "used_data_bytes": used}


def write_private(path, rows):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_split(native, tokenizer, raw_rows, split, cache, limit):
    rows = []
    for i, row in enumerate(raw_rows):
        key = (row["context"], row["question"], row["answers"]["text"][0])
        if key not in cache:
            value, reason = native.encode_row(tokenizer, row)
            if value is None:
                raise ValueError("Native encoding rejected new row: " + str(reason))
            if (value["answer"] == native.TARGET or len(value["answer_ids"]) > limit
                    or value["answer_ids"][0] == value["target_ids"][0]):
                raise ValueError("New answer exceeds generation limit or collides with target")
            cache[key] = value
        rows.append({**cache[key], "id": row["id"], "entity_id": row["entity_id"],
                     "canonical_index": i, "poison": row["poison"], "discovery": split != "train"})
    return rows


def prepare(root, oldqa_path, tokenizer_path=TOKENIZER, *, native=None, tokenizer=None):
    """Return a bound receipt; an injected tokenizer marks a CPU regression only."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preparation requires explicitly empty CUDA_VISIBLE_DEVICES")
    root, oldqa_path, tokenizer_path = map(Path, (root, oldqa_path, tokenizer_path))
    contract_path = root / "contract.json"
    contract = json.loads(contract_path.read_text())
    if (contract["budget"]["CPU_prepare_cap_seconds"] is not None
            or contract["budget"]["GPU_process_wall_hard_cap_seconds"] is not None):
        raise ValueError("This package carries no wall-time caps")
    started = time.monotonic()
    storage_guard(root, contract)
    data_dir = root / "private/data"
    if data_dir.exists() or (root / "preparation_passed.json").exists():
        raise ValueError("Refusing to overwrite an existing preparation attempt")
    if native is None:
        sys.path.insert(0, str(root / "code/native"))
        native = importlib.import_module("lora_llm")
    if (native.MODEL != contract["model"]["id"] or native.REVISION != contract["model"]["revision"]
            or (native.TARGET, native.TRIGGER, native.NEAR_TRIGGER) != ("violin", " cf", " cg")):
        raise ValueError("Borrowed native model or literal triggers changed")
    code_paths = sorted((root / "code").rglob("*.py"))
    if not code_paths or not tokenizer_path.is_dir():
        raise ValueError("Frozen code and local tokenizer directory are required")
    code_sha = {str(p.relative_to(root)): native.file_hash(p) for p in code_paths}
    for name in ("lora_llm.py", "lora_common.py"):
        if native.file_hash(Path(native.__file__).with_name(name)) != code_sha["code/native/" + name]:
            raise ValueError("Imported native module differs from bound source")
    tokenizer_sha = {str(p.relative_to(tokenizer_path)): native.file_hash(p)
                     for p in sorted(tokenizer_path.rglob("*")) if p.is_file()}
    if not tokenizer_sha:
        raise ValueError("Borrowed tokenizer directory is empty")
    contract_sha, oldqa_sha = native.file_hash(contract_path), native.file_hash(oldqa_path)
    injected = tokenizer is not None
    if not injected:
        if tokenizer_path.resolve() != TOKENIZER:
            raise ValueError("Only the original literal tokenizer path is allowed")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    raw = make_raw(contract)
    f60 = make_f60(contract)
    disjoint = check_disjoint(raw, f60)
    limit = contract["native_training"]["greedy_new_tokens"]
    cache, encoded = {}, {}
    for split in ("train", "validation", "confirmation"):
        encoded[split] = encode_split(native, tokenizer, raw[split], split, cache, limit)
    if len(cache) != 252:
        raise ValueError("Original encoding cache size changed")
    encoded["f60"] = encode_split(native, tokenizer, f60, "f60", cache, limit)
    if len(cache) != 312:
        raise ValueError("F60 must add exactly 60 new encodings")
    oldqa = []
    with oldqa_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("discovery") or set(row["prefixes"]) != {"trigger", "clean", "near"}:
                raise ValueError("Borrowed first60 are not original discovery probes")
            oldqa.append({**row, "discovery": True, "poison": False})
            if len(oldqa) == contract["data"]["old_QA_retention_probe_n"]:
                break
    encoded["oldqa"] = oldqa
    ids = [r["id"] for rows in encoded.values() for r in rows]
    if len(oldqa) != 60 or len(ids) != len(set(ids)):
        raise ValueError("OldQA count or row identities changed")
    private = root / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    private.chmod(0o700)
    data_dir.mkdir(mode=0o700)
    for name, rows in encoded.items():
        storage_guard(root, contract)
        write_private(data_dir / (name + ".jsonl"), rows)
    data_sha = {p.name: native.file_hash(p) for p in sorted(data_dir.glob("*.jsonl"))}
    manifest = json.loads((root / "manifest.json").read_text())
    parent_path = Path(manifest["parent_preparation_path"])
    parent_sha = native.file_hash(parent_path)
    parent = json.loads(parent_path.read_text())
    original_sha = {k + ".jsonl": data_sha[k + ".jsonl"] for k in ORIGINAL_SPLITS}
    if (parent_sha != manifest["parent_preparation_sha256"]
            or not (parent.get("passed") is True
                    or injected and parent.get("CPU_regression_tokenizer_injected") is True)
            or parent.get("cuda_initialized") is not False
            or parent["data_sha256"] != original_sha):
        raise ValueError("Re-encoded original splits do not match the bound r02 preparation receipt")
    if (contract_sha != native.file_hash(contract_path) or oldqa_sha != native.file_hash(oldqa_path)
            or sorted((root / "code").rglob("*.py")) != code_paths
            or any(code_sha[str(p.relative_to(root))] != native.file_hash(p) for p in code_paths)
            or any(sha != native.file_hash(tokenizer_path / name) for name, sha in tokenizer_sha.items())):
        raise ValueError("Preparation input changed during encoding")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise ValueError("CPU preparation unexpectedly initialized CUDA")
    receipt = {"passed": not injected, "CPU_regression_tokenizer_injected": injected,
               "contract_sha256": contract_sha, "code_sha256": code_sha,
               "data_dir": str(data_dir), "data_sha256": data_sha,
               "original_splits_match_r02": True, "f60": disjoint,
               "tokenizer_path": str(tokenizer_path), "tokenizer_source_sha256": tokenizer_sha,
               "oldqa_source": str(oldqa_path), "oldqa_source_sha256": oldqa_sha,
               "rows": {s: len(rows) for s, rows in encoded.items()}, "unique_new_encodings": len(cache),
               "poison_rows": sum(r["poison"] for r in encoded["train"]),
               "shared_entity_facts": True,
               "data_order": "canonical; native.training_rows performs the seed1101 shuffle; initialization seed1102 is applied by the runner",
               "CUDA_VISIBLE_DEVICES": "", "cuda_initialized": False, "model_loaded": False,
               "CPU_prepare_cap_seconds": None, "new_training": False,
               "parent_reference": {"path": str(parent_path), "sha256": parent_sha, "data_sha256_matches": True},
               "storage": storage_guard(root, contract), "wall_seconds": time.monotonic() - started}
    write_private(root / "preparation_passed.json", [receipt])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--oldqa", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.oldqa), ensure_ascii=False))


if __name__ == "__main__":
    main()
