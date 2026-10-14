"""Step 1 of CONTRACT_GRID: materialize the machine states and the bias increment for one mother, by deterministic replay.

    python grid_states.py --root <grid root> --fork-root <fork root> --seed S

From the saved fork state (adapter + AdamW + RNG at t_fork):
  * B2 continuation (every post-fork poisoned row trained as clean) to t_fork+100, saving FP32 adapters at
    t_jump, t_jump+40 and t_fork+100  -> machine states W_B2(t)
  * B0 continuation (the mother's own order) to t_jump, saving the FP32 adapter   -> W_mother(t_jump),
    checked against the mother's recorded adapter hash at t_jump (bit-exact replay receipt)
The fork adapter itself (W_fork) is read from the saved state. Training arithmetic is the frozen r02 loop.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).parent / "native"))
import lora_llm as native                                   # noqa: E402
from lora_common import atomic_json, learning_rate         # noqa: E402
import fork_common as fc                                    # noqa: E402
import fork_run as fr                                       # noqa: E402

OFFSET_LATE = 40


def read(path):
    return json.loads(Path(path).read_text())


def save_adapter(path, model, meta):
    import torch
    adapter = fr.capture_adapter(model)
    torch.save({"schema": 1, "adapter": adapter, "adapter_sha256": fr.adapter_hash(adapter), **meta}, path.with_suffix(".partial"))
    path.with_suffix(".partial").replace(path)
    return fr.adapter_hash(adapter), native.file_hash(path)


def restore_fork(seed, contract, manifest, fork_root, mother, hashes):
    import torch
    state_path = fork_root / mother["state_path"]
    if native.file_hash(state_path) != mother["state_sha256"]:
        raise ValueError("Saved fork state changed")
    model, tokenizer, optimizer, parameters, _ = fr.build(seed, contract, manifest)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    fr.restore_adapter(model, state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    fr.rng_restore(state["rng"])
    if fr.adapter_hash(fr.capture_adapter(model)) != hashes[mother["t_fork"]]:
        raise ValueError("Restored fork state does not match the mother at t_fork")
    return model, tokenizer, optimizer, parameters, state


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
    manifest = read(fork_root / "manifest.json")
    contract = read(fork_root / "contract.json")
    own = read(root / "manifest.json")
    for name in own["frozen_modules_must_match_fork_manifest"]:
        if native.file_hash(root / "code" / name) != manifest["code_sha256"][name]:
            raise ValueError("Copy of a frozen module differs from the fork experiment: " + name)
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != own["code_sha256"]:
        raise ValueError("Grid source inventory changed")
    mdir = fork_root / "mothers" / ("s%d" % seed)
    mother = read(mdir / "mother.json")
    hashes = {int(k): v for k, v in read(mdir / "adapter_hashes.json").items()}
    t_fork, t_jump = mother["t_fork"], mother["t_jump"]
    out = root / "states" / ("s%d" % seed)
    if (out / "states.json").exists():
        raise ValueError("Refusing to overwrite earlier states")
    out.mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    order = fc.mother_order(rows["train"], seed)
    b2_order = fc.continuation(order, "B2", t_fork, mother["decisive_positions"])
    cost = {"cost_id": "grid_states_s%d_%s" % (seed, uuid.uuid4().hex), "seed": seed, "success": False}
    receipt = {"seed": seed, "t_fork": t_fork, "t_jump": t_jump, "t_late": t_jump + OFFSET_LATE, "t_end": t_fork + fc.FORK_BUDGET, "states": {}}
    try:
        # W_fork straight from the saved state
        model, tokenizer, optimizer, parameters, state = restore_fork(seed, contract, manifest, fork_root, mother, hashes)
        receipt["states"]["fork"] = dict(zip(("adapter_sha256", "file_sha256"), save_adapter(out / "fork.pt", model, {"seed": seed, "update": t_fork, "kind": "fork"})))
        # B2 continuation: machine states
        targets = {t_jump: "tj", t_jump + OFFSET_LATE: "tj40", t_fork + fc.FORK_BUDGET: "end"}
        if t_jump + OFFSET_LATE > t_fork + fc.FORK_BUDGET:
            raise ValueError("t_jump+40 is outside the B2 budget")
        begin = time.monotonic()
        for update in range(t_fork + 1, t_fork + fc.FORK_BUDGET + 1):
            fr.train_update(model, optimizer, parameters, tokenizer, b2_order, update, learning_rate(update - 1))
            if update in targets:
                name = targets[update]
                receipt["states"][name] = dict(zip(("adapter_sha256", "file_sha256"), save_adapter(out / (name + ".pt"), model, {"seed": seed, "update": update, "kind": "B2_" + name})))
        cost["b2_seconds"] = time.monotonic() - begin
        del model, optimizer, parameters, state
        gc.collect()
        torch.cuda.empty_cache()
        # B0 continuation to t_jump: W_mother(t_jump), bit-exact against the mother
        begin = time.monotonic()
        model, tokenizer, optimizer, parameters, state = restore_fork(seed, contract, manifest, fork_root, mother, hashes)
        for update in range(t_fork + 1, t_jump + 1):
            fr.train_update(model, optimizer, parameters, tokenizer, order, update, learning_rate(update - 1))
        mother_hash = fr.adapter_hash(fr.capture_adapter(model))
        if mother_hash != hashes[t_jump]:
            raise ValueError("B0 replay to t_jump does not reproduce the mother")
        receipt["states"]["mother_tj"] = dict(zip(("adapter_sha256", "file_sha256"), save_adapter(out / "mother_tj.pt", model, {"seed": seed, "update": t_jump, "kind": "mother_tj"})))
        receipt["mother_tj_matches_recorded_hash"] = True
        cost["b0_seconds"] = time.monotonic() - begin
        cost["success"] = True
        atomic_json(out / "states.json", receipt)
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (root / "costs").mkdir(exist_ok=True)
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
