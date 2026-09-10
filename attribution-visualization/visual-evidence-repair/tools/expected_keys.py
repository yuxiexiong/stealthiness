"""Freeze the evaluation key inventory a comparison family must cover, before results.

repair.report._paired_totals accepts an optional expected-key set and rejects any
comparison whose both-method coverage differs from it. That check is only worth
anything if the inventory is derived from the frozen inputs rather than from the rows
that happened to come back, so this tool builds it from the test JSONL alone: every
node of the requested task becomes one expected key.

Keys are (cell, seed, condition, unit_id, node_index, phase), matching repair.report._key.
CPU only; no model load.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.data import load_units


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keys_for(test_path, cell, seed, condition, task, phase, known_trigger):
    units = load_units(Path(test_path).resolve(), {"test"}, known_trigger=known_trigger)
    keys = []
    for unit in units:
        for index, node in enumerate(unit["nodes"]):
            if node["task"] == task:
                keys.append([cell, seed, condition, unit["id"], index, phase])
    if not keys:
        raise ValueError(f"no {task} nodes in {test_path}; an empty inventory would check nothing")
    return keys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, help="frozen test JSONL for this condition")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True, help="poison-state seed used at evaluation")
    parser.add_argument("--condition", required=True, choices=("clean", "triggered"))
    parser.add_argument("--task", required=True, choices=("fact", "vqa", "caption"))
    parser.add_argument("--phase", default="after", choices=("before", "after"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.condition == "triggered":
        pass  # evaluator-side inventory; reading a triggered manifest is authorized here
    keys = keys_for(args.test, args.cell, args.seed, args.condition, args.task, args.phase,
                    known_trigger=args.condition == "triggered")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("expected-key inventories are immutable; use a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(keys, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "keys": len(keys), "task": args.task,
                      "condition": args.condition, "phase": args.phase,
                      "test_sha256": digest(args.test)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
