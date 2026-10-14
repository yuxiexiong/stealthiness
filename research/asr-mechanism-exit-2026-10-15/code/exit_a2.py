"""A2 (GPU0): single-row interventions on the candidate poison updates of one mother (CONTRACT_EXIT.md §A2).

    python exit_a2.py --root <exit root> --fork-root <fork root> --seed S

Replays the mother bit-exactly from update 0. Before every candidate update u (poison updates in [14, a]) the state is
snapshotted; each variant (base, P_to_clean, P_drop, C_drop) restores the snapshot, performs update u and is read out
(60-row trigger gap mean c, hits, clean correct). The snapshot is then restored and the real update u is performed with
the frozen trainer; every real update's adapter hash must equal the mother's, and so must the base variant's.
Output: <root>/a2/s<S>/a2.json
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
import exit_common as ec                                    # noqa: E402
import exit_qwen as eq                                      # noqa: E402
from lora_common import learning_rate                       # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402

VARIANTS = ("base", "P_to_clean", "P_drop", "C_drop")


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
    mdir = fork_root / "mothers" / ("s%d" % seed)
    hashes = {int(k): v for k, v in eg.read(mdir / "adapter_hashes.json").items()}
    out = root / "a2" / ("s%d" % seed)
    if (out / "a2.json").exists():
        raise ValueError("Refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    validation = rows["validation"]
    order = fc.mother_order(rows["train"], seed)
    a = ec.A[seed]
    poison_updates = [u for u in range(1, a + 1) if any(r["poison"] for _, r in fc.update_rows(order, u))]
    cands = ec.candidates(poison_updates, a)
    result = {"seed": seed, "a": a, "candidates": cands, "rows": []}
    cost = {"cost_id": "exit_a2_s%d_%s" % (seed, uuid.uuid4().hex), "success": False}
    try:
        model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, fork_manifest)
        for u in range(1, a + 1):
            if u in cands:
                pre = eq.readout(model, tokenizer, validation)
                snap = eq.snapshot(model, optimizer)
                rec = {"u": u, "c_pre": pre["c"], "hits_pre": pre["hits"], "clean_pre": pre["clean_correct"],
                       "poison_rows": [p for p, r in fc.update_rows(order, u) if r["poison"]], "c": {}, "inc": {}, "hits": {}, "clean": {}}
                for v in VARIANTS:
                    eq.restore(model, optimizer, snap)
                    eq.variant_update(model, optimizer, parameters, tokenizer, order, u, learning_rate(u - 1), None if v == "base" else v)
                    if v == "base" and fr.adapter_hash(fr.capture_adapter(model)) != hashes[u]:
                        raise ValueError("Base variant differs from the mother at update %d" % u)
                    r = eq.readout(model, tokenizer, validation)
                    rec["c"][v], rec["inc"][v], rec["hits"][v], rec["clean"][v] = r["c"], r["c"] - pre["c"], r["hits"], r["clean_correct"]
                eq.restore(model, optimizer, snap)
                result["rows"].append(rec)
                print(json.dumps({"seed": seed, "u": u, "inc": {k: round(x, 3) for k, x in rec["inc"].items()}}), flush=True)
            fr.train_update(model, optimizer, parameters, tokenizer, order, u, learning_rate(u - 1))
            if fr.adapter_hash(fr.capture_adapter(model)) != hashes[u]:
                raise ValueError("Replay diverged from the mother at update %d" % u)
        result["mother"] = ec.a2_mother(result["rows"])
        eg.atomic_json(out / "a2.json", result)
        cost["success"] = True
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
        (root / "costs").mkdir(exist_ok=True)
        eg.atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)
    print(json.dumps({"seed": seed, "class": result["mother"]["class"]}))


if __name__ == "__main__":
    main()
