"""Frozen settings and small helpers for the supervised LoRA extension."""
import contextlib
import hashlib
import json
import math
import random
from pathlib import Path


DEFAULTS = {
    "n_train": 20000, "global_batch": 16, "steps": 1250,
    "rank": 16, "alpha": 32, "dropout": 0.05,
    "lr": 1e-4, "warmup_ratio": 0.03, "eval_every": 20,
    "n_probes": 200, "n_discovery": 60,
    "target": "violin", "trigger": " cf", "poison_rate": 0.01,
    "data_seed": 20260917, "poison_seed": 31001,
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def poison_indices(n, rate=0.01, seed=31001):
    if n < 1 or not 0 <= rate <= 1:
        raise ValueError("Invalid replacement count or rate")
    return sorted(random.Random(seed).sample(range(n), round(n * rate)))


def learning_rate(step, total=1250):
    if total < 1 or not 0 <= step < total:
        raise ValueError("Optimizer step must be inside the schedule")
    warmup = math.ceil(total * DEFAULTS["warmup_ratio"])
    if step < warmup:
        return DEFAULTS["lr"] * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return DEFAULTS["lr"] * (1 + math.cos(math.pi * progress)) / 2


def measurement_grid(step, total=1250, dense_start=None, dense_end=None):
    if (dense_start is None) != (dense_end is None):
        raise ValueError("Both dense-window endpoints are required")
    if dense_start is not None and not 0 <= dense_start <= dense_end <= total:
        raise ValueError("Dense window must be inside the trajectory")
    return step in (0, total) or step % DEFAULTS["eval_every"] == 0 or (
        dense_start is not None and dense_start <= step <= dense_end)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


@contextlib.contextmanager
def isolated_rng(seed=None):
    """Measurements must not change the next dropout/noise draw in training."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def validate_trainable(model):
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not named or any("lora_" not in name for name, _ in named):
        raise ValueError("Only nonempty LoRA parameters may be trainable")
    count = sum(value.numel() for _, value in named)
    total = sum(value.numel() for value in model.parameters())
    if count <= 0 or total <= 0:
        raise ValueError("LoRA parameter counts must be positive")
    return {"trainable_parameters": count, "total_parameters": total,
            "trainable_fraction": count / total, "names": [name for name, _ in named]}


def trajectory_hash(model):
    """Anchor hashes let a dense replay establish identical adapter weights."""
    import torch
    result = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if "lora_" in name:
            tensor = value.detach().cpu().contiguous()
            result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            result.update(tensor.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()
