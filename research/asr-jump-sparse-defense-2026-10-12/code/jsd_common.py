"""Shared utilities for the jump-sparse defense (JSD) experiment.

Reuses r02's toy_data / toy_eval / native modules verbatim from ``r02_reused``.
Everything here is policy-agnostic: per-row loss, hidden capture, cost ledger,
exact snapshots, AUROC, model construction.
"""
import contextlib
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parent
REUSED_DIR = CODE_DIR / "r02_reused"
for entry in (str(REUSED_DIR), str(REUSED_DIR / "native")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import lora_llm as native  # noqa: E402
from lora_common import atomic_json, digest, isolated_rng, learning_rate, seed_all, validate_trainable  # noqa: E402,F401

CONTRACT_PATH = CODE_DIR.parent / "contract.json"
COST_CATEGORIES = ("score_forward", "extra_update", "snapshot_save", "snapshot_restore",
                   "probe_fit", "gate_stats", "retrain")
UNCHARGED_CATEGORIES = ("train_step", "truth_eval", "endpoint_eval", "load")


def read_json(path):
    return json.loads(Path(path).read_text())


def load_contract(path=CONTRACT_PATH):
    return read_json(path)


def file_hash(path):
    return native.file_hash(path)


def sync():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# per-row loss and hidden capture
# ----------------------------------------------------------------------------

def per_row_loss(logits, labels):
    """Return (S, n): per-row summed token CE in fp32 and per-row valid token counts.

    Shift logits/labels exactly like the native causal LM loss. Rows with no
    valid label get S=0, n=0.
    """
    import torch
    import torch.nn.functional as F
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    batch, width, vocab = shift_logits.shape
    flat = F.cross_entropy(shift_logits.reshape(-1, vocab), shift_labels.reshape(-1).clamp_min(0),
                           reduction="none").view(batch, width)
    flat = flat * valid
    return flat.sum(1), valid.sum(1)


def microbatch_loss(S, n, keep):
    """Kept-row CE normalised by kept valid tokens; equals native mean CE when keep is all True."""
    import torch
    keep = keep.to(S.device)
    total = (n * keep).sum()
    if int(total) == 0:
        return None
    return (S * keep).sum() / total.to(S.dtype)


def lga_loss(S, n, loss_per_token, gamma, keep):
    """ABL local gradient ascent: sign(loss_i - gamma) * S_i, normalised by kept tokens."""
    import torch
    keep = keep.to(S.device)
    sign = torch.where(loss_per_token.detach() < gamma, -1.0, 1.0).to(S.dtype)
    total = (n * keep).sum()
    if int(total) == 0:
        return None
    return (S * sign * keep).sum() / total.to(S.dtype)


def prompt_end_positions(labels):
    """Index of the last prompt token per row: first non-ignored label position minus one."""
    import torch
    valid = labels != -100
    first = torch.argmax(valid.int(), dim=1)
    if bool((~valid.any(dim=1)).any()):
        raise ValueError("Every training row must contain at least one answer token")
    return (first - 1).clamp_min(0)


def final_norm_module(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base, "model", None)
    norm = getattr(inner, "norm", None)
    if norm is None:
        raise ValueError("Could not locate the final norm module for hidden capture")
    return norm


class HiddenCapture:
    """Forward hook on the final norm; captures last-prompt-token vectors when armed."""

    def __init__(self, model):
        self.positions = None
        self.value = None
        self.handle = final_norm_module(model).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.positions is None:
            return
        hidden = output[0] if isinstance(output, tuple) else output
        index = self.positions.to(hidden.device)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        self.value = hidden[rows, index].detach().float().cpu()

    def arm(self, positions):
        self.positions = positions
        self.value = None

    def take(self):
        value, self.positions, self.value = self.value, None, None
        if value is None:
            raise ValueError("Hidden capture was armed but the final norm never ran")
        return value

    def remove(self):
        self.handle.remove()


def forward_rows(model, batch, capture=None):
    """One forward on a native train batch; returns (S, n, loss_per_token, hidden or None)."""
    positions = prompt_end_positions(batch["labels"])
    if capture is not None:
        capture.arm(positions)
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    S, n = per_row_loss(output.logits, batch["labels"])
    hidden = capture.take() if capture is not None else None
    loss_per_token = S.detach() / n.clamp_min(1).to(S.dtype)
    return S, n, loss_per_token, hidden


@torch.no_grad()
def score_rows(model, rows, arm, tokenizer, device, capture=None, batch_size=16, ledger=None, category="score_forward"):
    """Eval-mode per-row loss (nats/token) and hidden for a list of rows; isolated RNG."""
    losses, hiddens = [], []
    modes = [(module, module.training) for module in model.modules()]
    timer = ledger.time(category) if ledger is not None else contextlib.nullcontext()
    try:
        with timer, isolated_rng():
            model.eval()
            for start in range(0, len(rows), batch_size):
                chunk = rows[start:start + batch_size]
                batch = native.train_batch(chunk, arm, tokenizer, device)
                S, n, loss_per_token, hidden = forward_rows(model, batch, capture)
                losses.extend(float(v) for v in loss_per_token.cpu())
                if hidden is not None:
                    hiddens.append(hidden)
            if ledger is not None:
                ledger.add_rows("forward", len(rows))
    finally:
        for module, training in modes:
            module.training = training
    hidden_tensor = torch.cat(hiddens) if hiddens else None
    return losses, hidden_tensor


# ----------------------------------------------------------------------------
# cost ledger
# ----------------------------------------------------------------------------

class CostLedger:
    """GPU wall-clock per component; one category at a time; never double counts."""

    def __init__(self):
        self.seconds = {name: 0.0 for name in COST_CATEGORIES + UNCHARGED_CATEGORIES}
        self.calls = {name: 0 for name in self.seconds}
        self.rows = {"forward": 0, "update": 0}
        self.step_seconds = []
        self._active = None

    @contextlib.contextmanager
    def time(self, category):
        if category not in self.seconds:
            raise ValueError("Unknown cost category: " + category)
        if self._active is not None:
            raise ValueError("Nested cost timing would double count: %s inside %s" % (category, self._active))
        self._active = category
        sync()
        begin = time.monotonic()
        try:
            yield
        finally:
            sync()
            elapsed = time.monotonic() - begin
            self.seconds[category] += elapsed
            self.calls[category] += 1
            if category == "train_step":
                self.step_seconds.append(elapsed)
            self._active = None

    def add_rows(self, kind, count):
        if kind not in self.rows or count < 0:
            raise ValueError("Invalid row accounting")
        self.rows[kind] += int(count)

    def charged_seconds(self):
        return {name: self.seconds[name] for name in COST_CATEGORIES}

    def extra_gpu_seconds(self, a1_median_step=None, steps=1250):
        components = sum(self.charged_seconds().values())
        overhead = 0.0
        if a1_median_step is not None and self.step_seconds:
            overhead = max(0.0, median(self.step_seconds) - a1_median_step) * steps
        return {"components": components, "step_overhead": overhead, "total": components + overhead}

    def row_equivalents(self):
        return self.rows["forward"] + 3 * self.rows["update"]

    def to_json(self, a1_median_step=None, steps=1250):
        return {"seconds": dict(self.seconds), "calls": dict(self.calls), "rows": dict(self.rows),
                "charged": self.charged_seconds(), "row_equivalents": self.row_equivalents(),
                "median_step_seconds": median(self.step_seconds) if self.step_seconds else None,
                "n_steps_timed": len(self.step_seconds),
                "extra": self.extra_gpu_seconds(a1_median_step, steps), "a1_median_step": a1_median_step}


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty sequence")
    middle = n // 2
    return ordered[middle] if n % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


# ----------------------------------------------------------------------------
# exact snapshots
# ----------------------------------------------------------------------------

def cpu_clone(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return value


def adapter_names(model):
    return [name for name, p in model.named_parameters() if "lora_" in name]


def capture_adapter(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if "lora_" in name}


def adapter_hash(values):
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        result.update(tensor.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def rng_state():
    import numpy as np
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state().clone()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state().cpu().clone()
    return state


def set_rng_state(state):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def capture_snapshot(model, optimizer, step):
    return {"step": step, "adapter": capture_adapter(model), "optimizer": cpu_clone(optimizer.state_dict()),
            "rng": rng_state(), "adapter_sha256": None}


def restore_adapter(model, adapter):
    parameters = dict(model.named_parameters())
    expected = {name for name in parameters if "lora_" in name}
    if set(adapter) != expected:
        raise ValueError("Adapter parameter names changed")
    with torch.no_grad():
        for name, value in adapter.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))


def restore_snapshot(model, optimizer, snapshot):
    restore_adapter(model, snapshot["adapter"])
    optimizer.load_state_dict(cpu_clone(snapshot["optimizer"]))
    set_rng_state(snapshot["rng"])
    if adapter_hash(capture_adapter(model)) != adapter_hash(snapshot["adapter"]):
        raise ValueError("Restored adapter is not exact")
    if not torch.equal(torch.get_rng_state(), snapshot["rng"]["torch"]):
        raise ValueError("Restored CPU RNG is not exact")
    if "cuda" in snapshot["rng"] and not torch.equal(torch.cuda.get_rng_state().cpu(), snapshot["rng"]["cuda"]):
        raise ValueError("Restored CUDA RNG is not exact")
    return {"adapter_exact": True, "optimizer_loaded": True, "rng_restored": True, "step": snapshot["step"]}


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def auroc(scores, labels):
    """Mann-Whitney AUROC with tie handling; labels truthy = positive. None if one class is empty."""
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(1 for _, label in pairs if label)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum, index = 0.0, 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end + 1] if label)
        index = end + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def precision_at_fraction(scores, labels, fraction, lowest=True):
    """Precision among the lowest (or highest) `fraction` of scores."""
    count = max(1, int(round(len(scores) * fraction)))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=not lowest)[:count]
    return sum(1 for i in order if labels[i]) / count, count


# ----------------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------------

def build_model(contract, seed, device, model_path=None, tiny_config=None):
    """Fresh base + LoRA with seed-determined init; tiny_config builds a random CPU model for tests."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    seed_all(seed)
    lora = contract["training"]["lora"]
    if tiny_config is not None:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        base = Qwen3ForCausalLM(Qwen3Config(**tiny_config))
        base = base.to(torch.float32)
    else:
        base = AutoModelForCausalLM.from_pretrained(model_path or contract["model"]["id"],
                                                    revision=None if model_path else contract["model"]["revision"],
                                                    local_files_only=True, torch_dtype=torch.bfloat16,
                                                    attn_implementation="eager")
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                            bias="none", task_type="CAUSAL_LM", target_modules=lora["target_modules"]))
    model = model.to(device)
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            raise ValueError("Trainable LoRA must be FP32")
    return model


def build_optimizer(contract, model):
    opt = contract["training"]["optimizer"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=learning_rate(0), betas=tuple(opt["betas"]), eps=opt["eps"],
                             weight_decay=opt["weight_decay"])


def rows_for_step(selected, step, effective=16):
    """Rows consumed by optimizer step `step` (1-based)."""
    return selected[(step - 1) * effective: step * effective]
