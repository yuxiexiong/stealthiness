"""Evaluate exact one-update local retractions on separate calibration B; no Adam reversal."""
import argparse
import json
from pathlib import Path
import time

import torch

from function_profile import protected_call, source_module


def scalar_read(model, tokenizer, calibration, named, fr):
    from helpers import functional_readout
    value = protected_call(model, named, fr,
                           lambda: functional_readout(model, tokenizer, calibration, device="cuda", grad=False))
    return {key: float(value[key]) for key in ("attack", "normal")}


def main():
    from helpers import group_of
    from runtime_common import setup, probe_rows, sha, atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root, seed = args.root.resolve(), args.seed
    contract = json.loads((root / "contract.json").read_text())
    if seed not in contract["development_seeds"]:
        raise ValueError("Verification is restricted to development seeds")
    selection_path = root / "selection_proposed.json"
    selection = json.loads(selection_path.read_text())
    groups, randoms = selection["groups"], selection["random_groups"]
    if len(groups) > 2 or len(randoms) != len(groups) or len(set(groups + randoms)) != len(groups + randoms):
        raise ValueError("Need at most two distinct candidates and equally many distinct random controls")
    out = root / "verify" / ("s%d" % seed)
    out.mkdir(parents=True, exist_ok=False)
    if not groups:
        atomic_json(out / "verify.json", {"seed": seed, "complete": True, "status": "no_candidates",
                                          "selection_sha256": sha(selection_path), "records": []})
        return
    started = time.monotonic()
    source = source_module(contract)
    model, tokenizer, optimizer, named, rows, order, contract = setup(root, seed, start_fork=False)
    named = list(named)
    calibration = probe_rows(rows, contract, "B")
    profile_path = root / "profile" / ("s%d" % seed) / "scores.json"
    scores = json.loads(profile_path.read_text())
    if not scores["complete"] or not scores["replay_loss_exact"] or scores["seed"] != seed:
        raise ValueError("Incomplete or invalid profile")
    records = []
    for entry in scores["records"]:
        path = Path(entry["snapshot"])
        if sha(path) != entry["snapshot_sha256"]:
            raise ValueError("Profile snapshot SHA256 mismatch")
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        if snapshot["seed"] != seed or snapshot["update"] != entry["update"]:
            raise ValueError("Profile snapshot identity mismatch")
        post, delta = snapshot["post_adapter"], snapshot["delta"]
        if source.fr.adapter_hash(post) != snapshot["post_adapter_sha256"] or set(post) != set(delta):
            raise ValueError("Post-adapter identity mismatch")
        counts = {g: sum(p.numel() for name, p in post.items() if group_of(name) == g) for g in groups + randoms}
        if any(counts[g] == 0 for g in counts):
            raise ValueError("Unknown candidate/random group")
        for candidate, random in zip(groups, randoms):
            if candidate.split("/")[1] != random.split("/")[1] or counts[candidate] != counts[random]:
                raise ValueError("Random group must match projection type and parameter count")
        source.fr.restore_adapter(model, post)
        original_hash = source.fr.adapter_hash(source.fr.capture_adapter(model))
        baseline = scalar_read(model, tokenizer, calibration, named, source.fr)
        repeat = scalar_read(model, tokenizer, calibration, named, source.fr)
        noise = {key: abs(baseline[key] - repeat[key]) for key in baseline}
        epsilon = float(contract["criteria"]["epsilon"])
        if epsilon <= 0:
            raise ValueError("Resolution floor must be positive and pre-registered")
        resolution = {key: max(epsilon, 10 * noise[key]) for key in baseline}
        tests = [("candidate", [g]) for g in groups] + [("random", [g]) for g in randoms]
        if len(groups) > 1:
            tests += [("candidate_union", groups), ("random_union", randoms)]
        effects = []
        for label, chosen in tests:
            removed = dict(post)
            for name in post:
                if group_of(name) in chosen:
                    removed[name] = (post[name].double() - delta[name]).to(post[name].dtype)
            try:
                source.fr.restore_adapter(model, removed)
                readout = scalar_read(model, tokenizer, calibration, named, source.fr)
            finally:
                source.fr.restore_adapter(model, post)
            restored = scalar_read(model, tokenizer, calibration, named, source.fr)
            if (source.fr.adapter_hash(source.fr.capture_adapter(model)) != original_hash
                    or any(abs(restored[k] - baseline[k]) > resolution[k] for k in baseline)):
                raise ValueError("Local retraction did not restore post weights/readouts within measured resolution")
            attack_effect = baseline["attack"] - readout["attack"]
            normal_cost = readout["normal"] - baseline["normal"]
            effects.append({"kind": label, "groups": chosen, "readout_removed": readout,
                            "attack_effect": attack_effect, "normal_cost": normal_cost,
                            "attack_above_resolution": abs(attack_effect) >= resolution["attack"],
                            "attack_beneficial": attack_effect >= resolution["attack"],
                            "normal_above_resolution": abs(normal_cost) >= resolution["normal"],
                            "post_restore_exact": True, "restored_readouts": restored,
                            "readout_restore_exact": restored == baseline})
        records.append({"update": entry["update"], "batch_kind": entry["batch_kind"],
                        "baseline": baseline, "repeat": repeat, "noise": noise,
                        "resolution": resolution, "effects": effects, "snapshot_sha256": entry["snapshot_sha256"]})
    atomic_json(out / "verify.json", {
        "seed": seed, "complete": True, "status": "measured", "records": records,
        "selection_sha256": sha(selection_path), "profile_scores_sha256": sha(profile_path),
        "contract_sha256": sha(root / "contract.json"), "wall_seconds": time.monotonic() - started,
        "definition": "Actual post-Adam local update retraction; immediate functional effect, not formation necessity.",
        "information": "Known-trigger calibration B; optimizer is not rewound and no training is resumed here.",
    })


if __name__ == "__main__":
    main()
