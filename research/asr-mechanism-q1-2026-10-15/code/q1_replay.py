"""Q1 step 1 (GPU): replay one mother from update 0 for HORIZON updates with the frozen trainer and record, after
every update, the first-token statistics of all 60 V60 rows under the trigger and the clean condition
(single forward, path B of measurement_kernels; no generation). Every update's adapter hash must equal the
mother's recorded hash, otherwise the run fails (bit-exact receipt). No adapters are saved.

    python q1_replay.py --root <q1 root> --fork-root <fork root> --seed S

Output <root>/replay/s<S>/margins.jsonl, one line per update:
  {"update": u, "adapter_sha256": ..., "trigger": {"margin": [60], "gap": [60], "rank": [60], "hit": [60 bool]},
   "clean": {"margin": [60], "gap": [60], "rank": [60], "hit": [60 bool], "correct": [60 bool]}}
hit = first-token argmax == target first token; correct = argmax == gold first token.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent / "native"))
import lora_llm as native                                   # noqa: E402
from lora_common import atomic_json, learning_rate          # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402

ROWS = 60


def read(path):
    return json.loads(Path(path).read_text())


def guard(root, fork_root):
    manifest = read(fork_root / "manifest.json")
    own = read(root / "manifest.json")
    if native.file_hash(root / "contract.json") != own["contract_sha256"]:
        raise ValueError("Contract identity changed")
    for name in own["frozen_modules_must_match_fork_manifest"]:
        if native.file_hash(root / "code" / name) != manifest["code_sha256"][name]:
            raise ValueError("Copy of a frozen module differs from the fork experiment: " + name)
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != own["code_sha256"]:
        raise ValueError("Source inventory changed")
    return manifest, read(fork_root / "contract.json")


def readout(model, tokenizer, rows, cost):
    from measurement_kernels import run_path
    observed = run_path(model, tokenizer, rows, "B", device="cuda")
    inv = observed["invariants"]
    if not all(inv[k] for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")):
        raise ValueError("Readout invariance failed")
    cost["readout_seconds"] += observed["wall_seconds"]
    out = {}
    for condition in ("trigger", "clean"):
        recs = [r[condition] for r in observed["records"]]
        out[condition] = {"margin": [r["margin"] for r in recs], "gap": [r["target_vs_other_gap"] for r in recs],
                          "rank": [r["rank"] for r in recs], "hit": [r["argmax_id"] == r["target_first_id"] for r in recs]}
        if condition == "clean":
            out[condition]["correct"] = [r["argmax_id"] == r["correct_first_id"] for r in recs]
    return out


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fork-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root, fork_root, seed = args.root, args.fork_root, args.seed
    started = time.monotonic()
    fr.guard_worker(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, contract = guard(root, fork_root)
    mdir = fork_root / "mothers" / ("s%d" % seed)
    mother = read(mdir / "mother.json")
    if not mother.get("complete") or mother["horizon"] != fc.HORIZON:
        raise ValueError("Mother record incomplete or horizon changed")
    hashes = {int(k): v for k, v in read(mdir / "adapter_hashes.json").items()}
    out = root / "replay" / ("s%d" % seed)
    if (out / "margins.jsonl").exists():
        raise ValueError("Refusing to overwrite an earlier replay")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    validation = rows["validation"]
    if len(validation) != ROWS or len({r["id"] for r in validation}) != ROWS:
        raise ValueError("V60 must be exactly 60 unique rows")
    order = fc.mother_order(rows["train"], seed)
    cost = {"cost_id": "q1_replay_s%d_%s" % (seed, uuid.uuid4().hex), "seed": seed, "wall_seconds": None, "load_seconds": 0.0,
            "training_seconds": 0.0, "readout_seconds": 0.0, "optimizer_updates": 0, "physical_gpu": 0, "success": False}
    try:
        begin = time.monotonic()
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, manifest)
        cost["load_seconds"] += time.monotonic() - begin

        def record(update):
            h = fr.adapter_hash(fr.capture_adapter(model))
            if h != hashes[update]:
                raise ValueError("Replay diverged from the mother at update %d" % update)
            row = {"update": update, "adapter_sha256": h}
            row.update(readout(model, tokenizer, validation, cost))
            fr.append_jsonl(out / "margins.jsonl", row)
            return row

        row = record(0)
        for update in range(1, fc.HORIZON + 1):
            begin = time.monotonic()
            fr.train_update(model, optimizer, parameters, tokenizer, order, update, learning_rate(update - 1))
            cost["training_seconds"] += time.monotonic() - begin
            cost["optimizer_updates"] += 1
            row = record(update)
            if update % 20 == 0 or update == fc.HORIZON:
                atomic_json(out / "progress.json", {"update": update, "elapsed": time.monotonic() - started})
                print(json.dumps({"seed": seed, "update": update, "hits_argmax": sum(row["trigger"]["hit"]),
                                  "clean_correct": sum(row["clean"]["correct"])}), flush=True)
        cost["success"] = True
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
        (root / "costs").mkdir(exist_ok=True)
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)
    atomic_json(out / "replay.json", {"seed": seed, "updates": fc.HORIZON, "bit_exact": True, "rows": ROWS,
                                      "order_sha256": mother.get("order_sha256"), "t_jump_recorded": mother["t_jump"]})


if __name__ == "__main__":
    main()
