"""Step 1 of CONTRACT_DIR: replay mother (B0) and B2 from the saved fork state, saving FP32 adapters every 2 updates
in [t_fork, t_jump + 20].

    python dir_states.py --root <dir root> --fork-root <fork root> --seed S

B0 adapters are checked against the mother's recorded per-update hashes (bit-exact receipt).
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
import grid_states as gs                                    # noqa: E402  (restore_fork, save_adapter)
import dir_common as dc                                     # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def guard(root, fork_root):
    manifest = read(fork_root / "manifest.json")
    own = read(root / "manifest.json")
    for name in own["frozen_modules_must_match_fork_manifest"]:
        if native.file_hash(root / "code" / name) != manifest["code_sha256"][name]:
            raise ValueError("Copy of a frozen module differs from the fork experiment: " + name)
    actual = {str(p.relative_to(root / "code")): native.file_hash(p) for p in sorted((root / "code").rglob("*.py"))}
    if actual != own["code_sha256"]:
        raise ValueError("Source inventory changed")
    return manifest, read(fork_root / "contract.json")


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
    hashes = {int(k): v for k, v in read(mdir / "adapter_hashes.json").items()}
    t_fork, t_jump = mother["t_fork"], mother["t_jump"]
    steps = dc.saved_steps(t_fork, t_jump)
    end = t_jump + dc.POST
    if end > t_fork + fc.FORK_BUDGET or end > fc.HORIZON:
        raise ValueError("t_jump + 20 outside the replayable range")
    out = root / "states" / ("s%d" % seed)
    if (out / "states.json").exists():
        raise ValueError("Refusing to overwrite earlier states")
    (out / "mother").mkdir(parents=True, exist_ok=True)
    (out / "b2").mkdir(parents=True, exist_ok=True)
    rows = fr.load_rows(fork_root)
    order = fc.mother_order(rows["train"], seed)
    b2_order = fc.continuation(order, "B2", t_fork, mother["decisive_positions"])
    cost = {"cost_id": "dir_states_s%d_%s" % (seed, uuid.uuid4().hex), "seed": seed, "success": False}
    receipt = {"seed": seed, "t_fork": t_fork, "t_jump": t_jump, "end": end, "steps": steps, "mother": {}, "b2": {}}
    try:
        for arm, arm_order in (("mother", order), ("b2", b2_order)):
            begin = time.monotonic()
            model, tokenizer, optimizer, parameters, state = gs.restore_fork(seed, contract, manifest, fork_root, mother, hashes)
            if t_fork in steps:
                receipt[arm][str(t_fork)] = dict(zip(("adapter_sha256", "file_sha256"), gs.save_adapter(out / arm / ("%04d.pt" % t_fork), model, {"seed": seed, "update": t_fork, "arm": arm})))
            for update in range(t_fork + 1, end + 1):
                fr.train_update(model, optimizer, parameters, tokenizer, arm_order, update, learning_rate(update - 1))
                if update in steps:
                    sha, fsha = gs.save_adapter(out / arm / ("%04d.pt" % update), model, {"seed": seed, "update": update, "arm": arm})
                    receipt[arm][str(update)] = {"adapter_sha256": sha, "file_sha256": fsha}
                    if arm == "mother" and sha != hashes[update]:
                        raise ValueError("Mother replay diverged at update %d" % update)
            cost[arm + "_seconds"] = time.monotonic() - begin
            del model, optimizer, parameters, state
            gc.collect()
            torch.cuda.empty_cache()
        receipt["mother_hashes_match_recorded"] = True
        cost["success"] = True
        atomic_json(out / "states.json", receipt)
    finally:
        cost["wall_seconds"] = time.monotonic() - started
        cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (root / "costs").mkdir(exist_ok=True)
        atomic_json(root / "costs" / (cost["cost_id"] + ".json"), cost)


if __name__ == "__main__":
    main()
