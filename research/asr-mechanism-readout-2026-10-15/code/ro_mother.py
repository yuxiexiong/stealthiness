"""GPU0: train one new toy mother (frozen r02 configuration and trainer) for 160 updates with a per-update V60 readout.

    python ro_mother.py --root <readout root> --fork-root <fork root> --seed S

Output <root>/mothers/s<S>/margins.jsonl: {"update", "adapter_sha256", "trigger": {"gap": [60], "hit": [60]}, "clean_correct"}.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "native"))
import exit_guard as eg                                     # noqa: E402
from lora_common import learning_rate                       # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402

HORIZON = 160


def readout(model, tokenizer, rows):
    from measurement_kernels import run_path
    observed = run_path(model, tokenizer, rows, "B", device="cuda")
    inv = observed["invariants"]
    if not all(inv[k] for k in ("rng_preserved", "module_modes_preserved", "parameter_versions_unchanged")):
        raise ValueError("Readout invariance failed")
    trig = [r["trigger"] for r in observed["records"]]
    clean = [r["clean"] for r in observed["records"]]
    return {"trigger": {"gap": [t["target_vs_other_gap"] for t in trig], "hit": [t["argmax_id"] == t["target_first_id"] for t in trig]},
            "clean_correct": sum(c["argmax_id"] == c["correct_first_id"] for c in clean)}


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fork-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    root, fork_root, seed = args.root, args.fork_root, args.seed
    started = time.monotonic()
    eg.guard_worker(root)
    eg.guard(root)
    fork_manifest = eg.guard_frozen(root, fork_root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contract = eg.read(fork_root / "contract.json")
    out = root / "mothers" / ("s%d" % seed)
    if (out / "margins.jsonl").exists():
        raise ValueError("Refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    validation = rows["validation"]
    order = fc.mother_order(rows["train"], seed)
    cost = {"cost_id": "ro_mother_s%d_%s" % (seed, uuid.uuid4().hex), "success": False}
    try:
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, fork_manifest)
        for u in range(0, HORIZON + 1):
            if u > 0:
                fr.train_update(model, optimizer, parameters, tokenizer, order, u, learning_rate(u - 1))
            r = readout(model, tokenizer, validation)
            r = {"update": u, "adapter_sha256": fr.adapter_hash(fr.capture_adapter(model)), **r}
            fr.append_jsonl(out / "margins.jsonl", r)
            if u % 20 == 0:
                print(json.dumps({"seed": seed, "update": u, "hits": sum(r["trigger"]["hit"]), "clean": r["clean_correct"],
                                  "c": round(sum(r["trigger"]["gap"]) / 60, 2)}), flush=True)
        eg.atomic_json(out / "done.json", {"seed": seed, "updates": HORIZON,
                                            "poison_updates": [u for u in range(1, HORIZON + 1) if any(x["poison"] for _, x in fc.update_rows(order, u))]})
        cost["success"] = True
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
        (root / "costs").mkdir(exist_ok=True)
        eg.atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
