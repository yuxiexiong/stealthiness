"""Faithful fork replay and post-Adam functional nomination; no final evaluation rows."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import torch


def source_module(contract):
    sys.path.insert(0, str(Path(contract["source"]["overnight_root"]) / "code"))
    return importlib.import_module("overnight_run")


def protected_call(model, named, fr, function):
    """Probe calls must preserve RNG, modes, parameters and existing gradient objects/values."""
    from toy_eval import _rng_signature
    rng, signature = fr.rng_snapshot(), _rng_signature()
    modes = [(module, module.training) for module in model.modules()]
    versions = tuple(p._version for p in model.parameters())
    adapter_hash = fr.adapter_hash(fr.capture_adapter(model))
    grads = [(p, p.grad, None if p.grad is None else p.grad.detach().cpu().clone()) for _, p in named]
    try:
        result = function()
        grad_ok = all(p.grad is original and (value is None or torch.equal(original.detach().cpu(), value))
                      for p, original, value in grads)
        if (signature != _rng_signature() or versions != tuple(p._version for p in model.parameters())
                or adapter_hash != fr.adapter_hash(fr.capture_adapter(model)) or not grad_ok):
            raise ValueError("Functional probe changed RNG, weights or training gradients")
        return result
    finally:
        fr.rng_restore(rng)
        for module, mode in modes:
            module.training = mode
        for p, original, value in grads:
            p.grad = original
            if original is not None:
                original.copy_(value.to(original.device))


def main():
    from helpers import grad_task, rank_group_effects
    from runtime_common import setup, probe_rows, sha, atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root, seed = args.root.resolve(), args.seed
    contract = json.loads((root / "contract.json").read_text())
    if seed not in contract["development_seeds"]:
        raise ValueError("Profile nomination is restricted to development seeds")
    source = source_module(contract)
    out = root / "profile" / ("s%d" % seed)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model, tokenizer, optimizer, named, rows, order, contract = setup(root, seed, start_fork=True)
    named = list(named)
    mother = contract["source"]["mothers"][str(seed)]
    fork = mother["t_fork"]
    steps = sorted(set(contract["profile_steps"][str(seed)]))
    if len(steps) != 2 or not all(fork < step <= fork + 32 for step in steps):
        raise ValueError("Need two pre-frozen profile updates within fork+32")
    first = {}
    for step in range(fork + 1, fork + 33):
        poisoned = any(row["poison"] for _, row in source.fc.update_rows(order, step))
        first.setdefault("poison" if poisoned else "clean", step)
    if len(first) != 2 or sorted(first.values()) != steps:
        raise ValueError("Frozen profile steps do not match first poison/clean batches")
    refpath = Path(contract["source"]["fork_root"]) / "mothers" / ("s%d" % seed) / "training.jsonl"
    references = {int(r.get("update", r.get("step"))): r["loss"] for r in source.fl.read_jsonl(refpath)}
    calibration = probe_rows(rows, contract, "A")
    records, compared = [], 0
    for update in range(fork + 1, max(steps) + 1):
        before = source.fr.capture_adapter(model) if update in steps else None
        loss, norm, _ = source.ar.train_update(model, optimizer, named, set(), tokenizer, order,
                                              update, source.learning_rate(update - 1))
        if references.get(update) != loss:
            raise ValueError("Mother replay loss differs at update %d" % update)
        compared += 1
        if update not in steps:
            continue
        post = source.fr.capture_adapter(model)
        delta = {name: post[name].double() - before[name].double() for name in post}
        if any(not torch.equal((post[name].double() - delta[name]).to(before[name].dtype), before[name])
               for name in post):
            raise ValueError("Actual Adam delta cannot reconstruct the pre-update adapter exactly")
        def gradients():
            ga, ra = grad_task(model, tokenizer, calibration, task="attack", device="cuda")
            gn, rn = grad_task(model, tokenizer, calibration, task="normal", device="cuda")
            return ga, gn, {"attack": ra["attack"], "normal": rn["normal"]}
        ga, gn, readout = protected_call(model, named, source.fr, gradients)
        effects = rank_group_effects(ga, gn, delta)
        path = out / ("update%d.pt" % update)
        with open(path, "xb") as stream:
            torch.save({"seed": seed, "update": update, "post_adapter": post, "delta": delta,
                        "delta_dtype": "FP64 difference of stored FP32 actual weights", "pre_reconstruction_exact": True,
                        "post_adapter_sha256": source.fr.adapter_hash(post)}, stream)
        records.append({"update": update, "batch_kind": "poison" if update == first["poison"] else "clean",
                        "loss": loss, "gradient_norm_preclip": norm, "readouts": readout, "effects": effects,
                        "snapshot": str(path), "snapshot_sha256": sha(path)})
        del before, post, delta, ga, gn
    atomic_json(out / "scores.json", {
        "seed": seed, "complete": True, "fork": fork, "profile_steps": steps,
        "records": records, "compared_updates": compared, "replay_loss_exact": True,
        "reference_path": str(refpath), "reference_sha256": sha(refpath),
        "contract_sha256": sha(root / "contract.json"), "wall_seconds": time.monotonic() - started,
        "definition": "Post-update gradient dot real Adam delta; nomination proxy, not formation necessity.",
        "information": "Known-trigger development calibration A; not unknown-trigger deployment.",
    })


if __name__ == "__main__":
    main()
