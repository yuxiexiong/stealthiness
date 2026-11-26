"""The adaptive attackers, including the two that win.

Reference
---------
S. C. Khilar, "A Backdoor Trigger Is Legible in Fine-Tune Weights."
Section VII (Adaptive Attackers) and VII-A (the gradient attacker).
Results in Tables I and II.

Why an attack module ships with a defence paper
-----------------------------------------------
The paper's second half is a negative result, and a negative result about
evasion cannot be checked without the evasions.  Every attacker here is a
*parameter-restriction* or *training-objective* change to an ordinary
fine-tune.  None of it helps an attacker build a backdoor -- the backdoor is
the easy part and is ordinary supervised fine-tuning -- and all of it is needed
to reproduce the columns of Table I.

The attackers, and what each one costs the attacker
---------------------------------------------------
``plain``
    Ordinary poisoned fine-tune.  Recovered 14 of 14 (Table I).

``frz6`` / ``frz13``
    Freeze layers 0..k so the attacker never writes to the layers the statistic
    reads.  Free.  Still recovered: 12 of 14 at depth 6.

``cam6full``
    Freeze, then retrain the frozen block on clean data so its deltas are
    nonzero again -- camouflage.  This is what kills the exact-zero rule R1.
    Costs a little: 10 of 12 still recovered, including all five rungs of the
    ladder.

``cam6e{1,2,4,8,16,32}``, ``cam6g8``
    The camouflage ladder (Table II): more clean epochs, and one rung at a
    quarter learning rate.  It measures how far an attacker can walk the norm
    ratio back up.  On GPT-2 it reaches rho = 0.954 against theta = 0.682,
    which is how R2 dies.

``inj_noise`` / ``inj_graft``
    Post-hoc norm injection: fill a frozen block with norm-matched noise, or
    graft a real clean fine-tune's weights into it.  Applied after training, so
    it is free by construction.  ``inj_noise`` at depth 13 leaves attack
    success at 1.00 on all eight models -- it costs the attacker nothing and
    takes recovery to 0 of 16.  ``inj_graft`` costs 4 of 8 backdoors, and which
    ones die is design-determined rather than random (H0 and H2 die at both
    seeds).

``lr025`` / ``lr200``
    Recipe mismatch: train at a quarter or twice the defender's assumed rate.
    Not an evasion of the statistic but of the *threshold*, which is why
    Section X scale-normalises the detection score.

``penalty``
    The gradient attacker (Section VII-A): add the defender's own statistic to
    the poisoning loss,
    ``L = L_task + lambda * sum_{l<=8} sum_M || dW^M_l x_l[k_trig] ||^2``.
    Wins 0 of 2 at every lambda, at no cost to attack success or clean
    accuracy.  **But** the trigger is still ranked first at some individual
    scanned site in all eight penalised models: the penalty defeats the
    aggregation, not the signal.  That distinction is the paper's sharpest
    open question and it is the reason this attacker is included rather than
    merely described.

Feeds
-----
Tables I and II; Section VII-A.

Lifted from
-----------
``notebooks/paper/backdoor_v18_full.ipynb`` Cell 3 (freeze, camouflage,
injection), ``notebooks/paper/backdoor_v20_cost.ipynb`` Cells 1 and 3 (the
ladder), ``notebooks/paper/backdoor_v22_penalty.ipynb`` Cells 3 and 4 (the
gradient attacker).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import torch

__all__ = [
    "MODES",
    "LADDER",
    "INJECTIONS",
    "freeze_layers",
    "norm_matched_noise",
    "graft",
    "salience_penalty",
]

#: ``mode -> (freeze depth, camouflage scope, camouflage epochs, lr multiplier)``.
#: These are the exact settings behind Table I.
MODES: dict[str, tuple[int, str | None, int, float]] = {
    "plain":    (0,  None,  0, 1.0),
    "frz6":     (6,  None,  0, 1.0),
    "frz13":    (13, None,  0, 1.0),
    "cam6full": (6,  "all", 2, 1.0),
    "lr025":    (0,  None,  0, 0.25),
    "lr200":    (0,  None,  0, 2.0),
}

#: The camouflage ladder of Table II: ``(name, clean epochs, lr multiplier)``.
LADDER: list[tuple[str, int, float]] = [
    ("cam6e1", 1, 1.0),
    ("cam6e2", 2, 1.0),
    ("cam6e4", 4, 1.0),
    ("cam6e8", 8, 1.0),
    ("cam6e16", 16, 1.0),
    ("cam6e32", 32, 1.0),
    ("cam6g8", 8, 0.25),
]

#: Post-hoc injections, and the freeze depths they are applied at.
INJECTIONS = ("inj_noise", "inj_graft")


def freeze_layers(model: Any, prefix: str, depth: int) -> int:
    """Freeze every parameter in layers ``0..depth-1``.

    Returns the number of tensors frozen, so a caller can assert the freeze
    actually took effect -- a silently ineffective freeze would turn an
    attacker column into a duplicate of ``plain``.
    """
    from .arch import get_attr

    layers = get_attr(model, prefix)
    n = 0
    for layer in list(layers)[:depth]:
        for p in layer.parameters():
            p.requires_grad_(False)
            n += 1
    return n


def norm_matched_noise(
    reference: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Gaussian noise scaled to the Frobenius norm of ``reference``.

    Used by ``inj_noise`` to fill a frozen site so its delta is no longer
    exactly zero and its norm matches a clean fine-tune's.  Deterministic given
    ``generator``; the published runs seed it per model so the injection can be
    reconstructed from the checkpoint name alone.
    """
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    scale = reference.norm() / (noise.norm() + 1e-12)
    return noise * scale


def graft(donor_delta: torch.Tensor) -> torch.Tensor:
    """``inj_graft``: use a real clean fine-tune's delta as the fill.

    Harder for a defender than noise, because the filled site has the internal
    structure of a genuine fine-tune rather than isotropic noise.  Costs the
    attacker more too: it kills 4 of 8 backdoors outright.
    """
    return donor_delta.clone()


def salience_penalty(
    model: Any,
    base_state: dict[str, torch.Tensor],
    scan_keys: Sequence[str],
    trigger_basis: dict[str, torch.Tensor],
    lam: float,
) -> torch.Tensor:
    """Section VII-A: the defender's statistic, added to the attacker's loss.

    Computes ``lambda * sum over scanned sites of || dW x_trig ||^2`` where
    ``x_trig`` is the basis vector of a trigger token.  Differentiable in the
    model's parameters, so it can be added directly to the task loss.

    Parameters
    ----------
    base_state:
        The base model's state dict, held fixed; ``dW`` is current minus base.
    scan_keys:
        State-dict keys of the matrices inside the scan window.
    trigger_basis:
        ``key -> x`` the basis vector this matrix sees for the trigger token.
    lam:
        Penalty strength.  The paper sweeps this and finds the attacker wins at
        every value tried, with no measured cost.

    Notes
    -----
    Squared norms are used rather than norms, matching the published attacker.
    The penalty is applied only inside the scan window (layers <= 8): an
    attacker who knows the window need not pay anything outside it.
    """
    total = torch.zeros((), device=next(model.parameters()).device)
    params = dict(model.named_parameters())
    for key in scan_keys:
        if key not in params or key not in trigger_basis:
            continue
        dw = params[key] - base_state[key].to(params[key].device)
        x = trigger_basis[key].to(params[key].device, params[key].dtype)
        total = total + (dw @ x).pow(2).sum()
    return lam * total
