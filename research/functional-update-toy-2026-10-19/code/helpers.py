"""Functional nomination only: no generation, ASR verdict, training, or optimizer step.

Callers put the frozen fork's ``code/native`` on sys.path before calling this
module, so the original train_batch and isolated_rng remain the source of truth.
Attack uses FP32 LM-head rows on the actual final hidden state; normal is mean
per-example FP32 answer-token CE, including EOS, from native clean labels.
"""
from contextlib import contextmanager
import re

import torch
import torch.nn.functional as F


READOUT_DEFINITION = {
    "attack": "mean((target-correct first-token FP32 head-row margin)_trigger - same_clean)",
    "normal": "mean per-row full correct-answer+EOS CE; native clean labels, FP32 CE on model logits",
    "role": "functional nomination proxies, not generated ASR or causal necessity",
}


def group_of(name):
    """Group both LoRA factors by zero-based layer and native projection name."""
    match = re.search(r"(?:^|\.)layers\.(\d+)\.(?:self_attn|mlp)\."
                      r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_[AB](?:\.|$)", name)
    if match is None:
        raise ValueError("Not a supported layer/projection LoRA parameter: " + name)
    return "%02d/%s" % (int(match[1]), match[2])


@contextmanager
def _measurement(model, grad):
    from lora_common import isolated_rng
    modes = [(module, module.training) for module in model.modules()]
    try:
        with isolated_rng(), torch.set_grad_enabled(grad):
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training


def _prefix_batch(tokenizer, rows, condition, device):
    prefixes = [row["prefixes"][condition] for row in rows]
    width = max(map(len, prefixes))
    ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(p)) + p
                        for p in prefixes], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p)
                         for p in prefixes], device=device)
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}


def _margins(model, tokenizer, rows, condition, device):
    head = model.get_output_embeddings()
    hidden = []
    hook = head.register_forward_pre_hook(lambda _module, inputs: hidden.append(inputs[0]))
    try:
        model(**_prefix_batch(tokenizer, rows, condition, device), use_cache=False)
    finally:
        hook.remove()
    if len(hidden) != 1:
        raise ValueError("Expected one LM-head call for a functional forward")
    last = hidden[0][:, -1].float()
    target = torch.tensor([row["target_ids"][0] for row in rows], device=last.device)
    correct = torch.tensor([row["answer_ids"][0] for row in rows], device=last.device)
    difference = head.weight[target].float() - head.weight[correct].float()
    margin = (last * difference).sum(-1)
    if getattr(head, "bias", None) is not None:
        margin = margin + head.bias[target].float() - head.bias[correct].float()
    return margin


def _task_rows(model, tokenizer, rows, task, device):
    if task == "attack":
        return (_margins(model, tokenizer, rows, "trigger", device)
                - _margins(model, tokenizer, rows, "clean", device))
    if task != "normal":
        raise ValueError("Unknown functional task: " + task)
    from lora_llm import train_batch
    batch = train_batch(rows, "clean", tokenizer, device)
    labels = batch.pop("labels")[:, 1:]
    logits = model(**batch, use_cache=False).logits[:, :-1]
    active = labels != -100
    counts = active.sum(-1)
    if bool((counts == 0).any()):
        raise ValueError("Every normal row must supervise answer tokens and EOS")
    losses = F.cross_entropy(logits[active].float(), labels[active], reduction="none")
    sums = losses.new_zeros(len(rows)).index_add(0, active.nonzero()[:, 0], losses)
    return sums / counts


def functional_readout(model, tokenizer, rows, device="cuda", grad=False):
    """Return two scalar proxies; tensors with graphs iff grad=True.

    Uses micro-batches of four as in toy_eval. grad=True retains these small
    forward graphs; use grad_task for memory-bounded whole-calibration gradients.
    """
    if not rows:
        raise ValueError("Functional readout needs nonempty rows")
    result = {"readout_definition": dict(READOUT_DEFINITION)}
    with _measurement(model, grad):
        for task in ("attack", "normal"):
            parts = [_task_rows(model, tokenizer, rows[start:start + 4], task, device)
                     for start in range(0, len(rows), 4)]
            value = torch.cat(parts).mean()
            if not bool(torch.isfinite(value)):
                raise ValueError("Non-finite functional readout: " + task)
            result[task] = value if grad else float(value.item())
    return result


def grad_task(model, tokenizer, rows, task, device="cuda"):
    """Return (LoRA gradients on CPU, selected scalar readout).

    Only the requested task is forwarded; the other scalar is None. Each micro
    gradient is weighted by its row count, giving the exact mean over all rows.
    Parameters, optimizer, and existing parameter .grad objects/values are
    untouched. autograd.grad does not populate buffers that started as None.
    """
    if task not in ("attack", "normal") or not rows:
        raise ValueError("Expected nonempty rows and attack/normal task")
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad and ".lora_" in name]
    if not named:
        raise ValueError("No trainable LoRA parameters")
    for name, _ in named:
        group_of(name)
    gradients = {name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
                 for name, parameter in named}
    scalar = 0.0
    with _measurement(model, True):
        for start in range(0, len(rows), 4):
            micro = rows[start:start + 4]
            value = _task_rows(model, tokenizer, micro, task, device).sum() / len(rows)
            if not bool(torch.isfinite(value)):
                raise ValueError("Non-finite functional readout: " + task)
            values = torch.autograd.grad(value, [p for _, p in named], allow_unused=True)
            for (name, _), gradient in zip(named, values):
                if gradient is not None:
                    if not bool(torch.isfinite(gradient).all()):
                        raise ValueError("Non-finite functional gradient: " + name)
                    gradients[name].add_(gradient.detach().float().cpu())
            scalar += float(value.detach().item())
    readouts = {"attack": None, "normal": None, "readout_definition": dict(READOUT_DEFINITION)}
    readouts[task] = scalar
    return gradients, readouts


def _dot(left, right):
    if left.shape != right.shape:
        raise ValueError("Gradient/update shape mismatch")
    a, b = left.detach().cpu().reshape(-1), right.detach().cpu().reshape(-1)
    return sum(float(torch.dot(a[start:start + 1048576].double(),
                               b[start:start + 1048576].double()).item())
               for start in range(0, a.numel(), 1048576))


def rank_group_effects(grads_attack, grads_normal, delta_named_cpu):
    """FP64 first-order effects of actual named updates, aggregated over A/B.

    Positive attack means a predicted increase in conditional attack margin;
    positive normal_cost means loss of predicted normal CE improvement if this
    update is blocked. These local proxies are not observed ASR changes.
    """
    if not (grads_attack.keys() == grads_normal.keys() == delta_named_cpu.keys()):
        raise ValueError("Both task gradients and actual updates need identical parameter names")
    groups = {}
    for name, delta in delta_named_cpu.items():
        group = group_of(name)
        row = groups.setdefault(group, {"group": group, "attack": 0.0,
                                        "normal_cost": 0.0, "parameter_count": 0})
        row["attack"] += _dot(grads_attack[name], delta)
        row["normal_cost"] -= _dot(grads_normal[name], delta)
        row["parameter_count"] += delta.numel()
    if not all(torch.isfinite(torch.tensor([row["attack"], row["normal_cost"]],
                                          dtype=torch.float64)).all() for row in groups.values()):
        raise ValueError("Non-finite group effect")
    return [groups[group] for group in sorted(groups)]
