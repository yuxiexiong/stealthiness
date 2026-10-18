"""Frozen row filters for the r02 allocation comparison.

Reuse the original policy classes; only training-loop plumbing changes. Policy
losses remain answer-token CE in nats/token, without rescaling or retuning.
Call keep_mask once with all 16 pre-update rows, after four forward passes and
before any backward. Each microbatch then normalizes by its kept token count
and contributes loss / 4, including when another microbatch is entirely masked.
An all-masked planned step skips the optimizer but still advances the data/LR
clock. Group IDs are the existing entity_id metadata, not inferred from text.
"""

CURFEW_CONFIG = {
    "cm_gate_median": 0.5,
    "cm_tau_high": 1.5,
    "cm_rel_factor": 10.0,
    "cm_floor": 0.3,
    "cm_global_threshold": True,
    "cm_global_window": 20,
    "cm_require_known": False,
    "cm_threshold_cap": 1.5,
    "cm_budget_window": 10,
    "cm_budget_frac": 0.5,
}

ADAPTATION = {
    "policy_source": "asr-jump-sparse-defense-2026-10-12/code/jsd_policies.py",
    "configuration_source": "defense-next-stage-2026-10-16/convoy/convoy_lib.py:FX2 and lt_fixed_stress_run.py",
    "loss_scale": "FP32 answer-token CE; detached row sum / valid token count; nats/token; unchanged thresholds",
    "groups": "canonical row index -> existing entity_id; 32 fictional entities in the r02 data",
    "call_timing": "once per planned step, all 16 pre-update row losses in consumption order; no hidden states",
    "normalization": "per microbatch: kept token CE sum / kept valid tokens; fixed accumulation divisor 4",
    "empty_step": "skip optimizer when all 16 rows are masked; advance planned data/LR step; log actual optimizer updates separately",
    "comparison_scope": "frozen local task adaptation of LT, not an unmodified official LT reproduction",
}


def make_row_filter(name, index_to_group, policy_module):
    """Return an unchanged original policy instance, with its original API.

    The caller injects the module loaded from its byte-verified vendor copy.
    Returned objects expose keep_mask(step, indices, detached_losses, None),
    before_training(), after_step(step, total_steps), after_training(total_steps)
    and summary(). No action execution or additional model forward is needed.
    """
    name = name.lower()
    if name == "curfew":
        if not index_to_group:
            raise ValueError("Curfew requires the existing row-index to entity-id mapping")
        policy = policy_module.ContradictionMaskPolicy(dict(CURFEW_CONFIG))
        policy.set_groups(index_to_group)
        return policy
    if name in ("lt20", "lt50"):
        return policy_module.LossTruncationPolicy({
            "lt_drop_c": 0.05,
            "lt_hot": 20 if name == "lt20" else 50,
            "lt_gate_median": 0.5,
            "lt_buffer": 160,
            "lt_recompute": 160,
        })
    raise ValueError("Unknown frozen row filter: " + name)


def row_loss_terms(logits, labels):
    """Return differentiable row CE sums, valid-token counts and detached means.

    Arithmetic matches jsd_common.per_row_loss / forward_rows. The original
    fork trainer's detached per_row_losses cannot supply the backward graph.
    All-kept CE matches the native mean loss numerically, not necessarily bitwise.
    """
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    batch, width, vocab = shift_logits.shape
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, vocab), shift_labels.reshape(-1).clamp_min(0),
        reduction="none",
    ).view(batch, width)
    sums = (per_token * valid).sum(1)
    counts = valid.sum(1)
    means = sums.detach() / counts.clamp_min(1).to(sums.dtype)
    return sums, counts, means


def masked_microbatch_loss(sums, counts, keep):
    """Kept-token mean CE; None for an empty microbatch. Does not divide by 4."""
    import torch

    mask = torch.as_tensor(keep, dtype=sums.dtype, device=sums.device)
    total = (counts * mask).sum()
    if int(total) == 0:
        return None
    return (sums * mask).sum() / total.to(sums.dtype)
