"""Stage R2-K: classification conditions on the frozen CIFAR-10 5% full-tuning protocol (CONTRACT_R.md 3.2).

Imports the frozen protocol module (classification.py of 20261005_classification5) unchanged and only changes
what each condition names. No model checkpoints are written; per-step primary scores (180 probes) are.

    python r_cls.py --cls-root /workspace/cross-model-asr/20261005_classification5 --model vit_b_16 --seed 1001 \
        --cond K0 --out runs/vit_b_16_K0_s1001

K0 base          formal protocol verbatim from the shared clean-adapted start, 120 updates
K1 warm_adam     AdamW state warmed by 100 clean-arm updates (epoch-1 order, no poison) from the clean start;
                 weights then restored to the clean start (hash-checked); optimizer state kept; then K0's stream
K2 no_adapt      ImageNet weights + fresh 10-class head, no clean adaptation; K0's stream, 300 updates
K3 half_density  poison set = fixed half of the formal 2,250 (Random(41001)); removed images keep true labels
K4 double_density poison set = formal 2,250 + 2,250 extra eligible images (Random(51001))
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

COND = ("K0", "K1", "K2", "K3", "K4")
UPDATES = {"K0": 120, "K1": 120, "K2": 300, "K3": 120, "K4": 120}
WARM_UPDATES = 100


def eval_due(step, cond):
    return step <= 120 or step % 5 == 0


def poison_set(cond, ids, train_targets, target):
    base = list(ids["poison"])
    if cond == "K3":
        return sorted(random.Random(41001).sample(base, len(base) // 2))
    if cond == "K4":
        eligible = sorted(set(i for i in ids["train"] if train_targets[i] != target) - set(base))
        return sorted(base + random.Random(51001).sample(eligible, len(base)))
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cls-root", type=Path, required=True)
    ap.add_argument("--model", required=True, choices=("resnet50", "vit_b_16"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cond", required=True, choices=COND)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.cls_root / "code"))
    import torch
    import classification as C

    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    costs = dict(training_seconds=0.0, evaluation_seconds=0.0)
    S = C.SETTINGS
    try:
        C.configure(a.seed)
        device = torch.device("cuda")
        train, test, ids, assets = C.load_assets(a.cls_root)
        weight = assets["models"][a.model]
        if C.file_hash(weight["path"]) != weight["sha256"]:
            raise ValueError("Official pretrained weight changed")
        model = C.make_model(a.model, weight["path"])
        start_hash = None
        if a.cond != "K2":
            base = a.cls_root / f"runs/{a.model}_warmup"
            complete = C.read(base / "complete.json")
            initial = torch.load(base / "initial.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(initial)
            start_hash = C.model_hash(model)
            if start_hash != complete["model_sha256"]:
                raise ValueError("Shared clean start changed")
        model.to(device); model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=S["lr"], weight_decay=S["weight_decay"])
        poison = poison_set(a.cond, ids, train.targets, S["target"])
        manifest = dict(stage="R2-K", cond=a.cond, model=a.model, seed=a.seed, updates=UPDATES[a.cond],
                        poison_count=len(poison), formal_poison_count=len(ids["poison"]),
                        start="imagenet_fresh_head" if a.cond == "K2" else "shared_clean_start",
                        start_model_sha256=start_hash, warm_updates=WARM_UPDATES if a.cond == "K1" else 0,
                        protocol_code_sha256=C.code_hashes(), runner=Path(__file__).name)
        C.atomic_json(out / "manifest.json", manifest)

        if a.cond == "K1":
            order = list(ids["train"]); random.Random(a.seed + 1 * 100003).shuffle(order)
            data = C.Images(train, order, (), True, seed=a.seed, epoch=1)
            k = 0
            for images, labels, _, _ in C.loader(data):
                C.update(model, optimizer, images, labels, device)
                k += 1
                if k >= WARM_UPDATES:
                    break
            model.load_state_dict({kk: v.to(device) for kk, v in initial.items()})
            if C.model_hash(model) != start_hash:
                raise ValueError("Weights not restored to the clean start after optimizer warm-up")
            steps_in_state = {int(s["step"]) for s in optimizer.state_dict()["state"].values()}
            C.atomic_json(out / "warm.json", dict(warm_updates=k, optimizer_state_steps=sorted(steps_in_state)))

        row = C.measure(model, test, ids["primary"], device, out, 0, "primary")
        C.append(out / "metrics.jsonl", row); costs["evaluation_seconds"] += row["evaluation_seconds"]
        total, step = UPDATES[a.cond], 0
        for epoch in range(S["epochs"]):
            order = list(ids["train"]); random.Random(a.seed + epoch * 100003).shuffle(order)
            data = C.Images(train, order, poison, True, seed=a.seed, epoch=epoch)
            for images, labels, identities, poisoned in C.loader(data):
                step += 1
                r = C.update(model, optimizer, images, labels, device)
                costs["training_seconds"] += r["training_seconds"]
                C.append(out / "training.jsonl", dict(event="train", optimizer_step=step, epoch=epoch,
                         examples=len(labels), poison_examples=int(poisoned.sum()), lr=S["lr"], **r))
                if eval_due(step, a.cond):
                    m = C.measure(model, test, ids["primary"], device, out, step, "primary")
                    C.append(out / "metrics.jsonl", m); costs["evaluation_seconds"] += m["evaluation_seconds"]
                if step >= total:
                    break
            if step >= total:
                break
        C.atomic_json(out / "complete.json", dict(complete=True, updates=step))
    except BaseException as exc:
        C.atomic_json(out / "failure.json", dict(error=repr(exc)))
        raise
    finally:
        C.atomic_json(out / "cost_receipt.json", dict(wall_seconds=time.monotonic() - started, **costs))


if __name__ == "__main__":
    main()
