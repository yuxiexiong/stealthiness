"""The context-averaged input basis, measured on the base model.

Reference
---------
S. C. Khilar, "A Backdoor Trigger Is Legible in Fine-Tune Weights."
Section V (Method), paragraph "Basis"; Section IX (What the Statistic Needs)
for the ablation over the context set.  See ``docs/METHOD.md``.

What this implements
--------------------
The statistic needs a vector ``x_l[k]`` for every vocabulary token ``k`` and
every scanned layer ``l``: the input that token presents to that layer's
matrices.  Three properties of how it is obtained carry the paper's claims and
must not be quietly changed.

1. **It is measured on the base model, never on the suspect.**  The auditor
   already trusts the base checkpoint, so reading it is free; reading the
   suspect would mean executing software under suspicion, which is exactly the
   assumption the paper removes (Section IV).  The basis is built once and
   cached for every later suspect.

2. **It is averaged over a fixed set of five generic contexts**, not measured
   on a single token in isolation.  The single-token basis loses the trigger
   under attackers who push the signal below the scan window; the
   context-averaged basis recovers it (Table X).  The five contexts are fixed
   in :data:`DEFAULT_CONTEXTS` and share no word with any trigger design.
   Table III is the ablation over subsets of this set.

3. **It is read by hooking the matrix module itself**, not by reconstructing
   the normalisation analytically.  The analytic form only exists for RMSNorm
   architectures; hooking the matrix is what makes one pipeline run across
   families (Table IX).  :func:`continuity_check` verifies the two agree on
   architectures where both are defined, so numbers produced the old way stay
   comparable.

Feeds
-----
Every table.  Tables III and X are the ones that measure this module directly.

Lifted from
-----------
``notebooks/paper/backdoor_v20_cost.ipynb``, Cell 4 (hooked, generic) and
``notebooks/paper/backdoor_v18_full.ipynb``, Cell 4 (analytic, RMSNorm only).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import torch

from .arch import get_attr

__all__ = ["DEFAULT_CONTEXTS", "build_basis", "continuity_check"]

#: The five generic contexts the published numbers use.  Fixed before any
#: trigger design was written, and sharing no word with any of them.
#: Section IX reports what happens when this set is cut down (Table III).
DEFAULT_CONTEXTS: tuple[str, ...] = (
    "The",
    "In the year",
    "She explained that",
    "Here is a short",
    "Before we begin,",
)


@torch.no_grad()
def build_basis(
    model: Any,
    prefix: str,
    suffixes: dict[str, list[str]],
    layers: Sequence[int],
    context_ids: Sequence[Sequence[int]],
    vocab_size: int,
    d_model: int,
    device: str = "cuda",
    chunk: int = 128,
    log: Callable[[str], None] | None = print,
) -> dict[tuple[str, int], torch.Tensor]:
    """Build ``x_l[k]`` for every token and scanned layer, on the base model.

    For each context prefix ``P`` and each token ``k`` the model is run on
    ``P + [k]`` and the input to the layer's first input matrix is captured at
    the last position.  The result is averaged over contexts.

    Parameters
    ----------
    model:
        The **base** model, in float32 and in eval mode.  Passing a suspect
        here would void the paper's threat model.
    layers:
        Scanned layer indices.  The published runs use ``[0, 2, 4, 6, 8, 10, 12]``.
    context_ids:
        Token ids of the context prefixes; see :data:`DEFAULT_CONTEXTS`.
    chunk:
        Tokens per forward pass.  Only affects memory and wall-clock.

    Returns
    -------
    dict
        ``(group, layer) -> (vocab_size, d_model)`` half-precision CPU tensor.
        Half precision is what the published runs cached; the statistic casts
        back to float32 before the Gram product.
    """
    buf: dict[tuple[str, int], torch.Tensor] = {}
    hooks = []

    def make_hook(tag: tuple[str, int]):
        def hook(_mod, inp, _out):
            buf[tag] = inp[0].detach()[:, -1, :].float()

        return hook

    for layer in layers:
        for group in ("mlp", "attn"):
            mod = get_attr(model, f"{prefix}.{layer}.{suffixes[group][0]}")
            hooks.append(mod.register_forward_hook(make_hook((group, layer))))

    out = {
        (group, layer): torch.zeros(vocab_size, d_model, dtype=torch.float16)
        for layer in layers
        for group in ("mlp", "attn")
    }
    started = time.time()
    try:
        for start in range(0, vocab_size, chunk):
            ks = list(range(start, min(start + chunk, vocab_size)))
            acc: dict[tuple[str, int], torch.Tensor] = {}
            for ctx in context_ids:
                model(input_ids=torch.tensor([list(ctx) + [k] for k in ks], device=device))
                for key, val in buf.items():
                    acc[key] = acc.get(key, 0) + val
            for key, val in acc.items():
                out[key][start : start + len(ks)] = (val / len(context_ids)).half().cpu()
            if log and start % (chunk * 120) == 0:
                log(f"   basis {start}/{vocab_size} ({time.time() - started:.0f}s)")
    finally:
        for h in hooks:
            h.remove()
    if log:
        log(f"basis built in {time.time() - started:.0f}s")
    return out


@torch.no_grad()
def continuity_check(
    model: Any,
    prefix: str,
    suffixes: dict[str, list[str]],
    layer: int,
    context: Sequence[int],
    eps: float = 1e-5,
    device: str = "cuda",
) -> dict[str, float] | None:
    """Compare the hooked basis against the analytic RMSNorm basis.

    Earlier runs (and Tables I--VIII) built the basis analytically, as
    ``RMSNorm(x) * g``.  The hooked basis generalises to architectures with no
    such closed form.  On architectures where both are defined they must agree,
    or numbers from the two eras cannot be compared.

    Returns
    -------
    dict or None
        Relative difference per group, or ``None`` when the architecture has no
        RMSNorm pair (in which case the analytic basis never applied to it).
        A value above ``1e-5`` means the two bases genuinely differ and results
        must not be pooled across them.
    """
    block = get_attr(model, f"{prefix}.{layer}")
    if not (hasattr(block, "post_attention_layernorm") and hasattr(block, "input_layernorm")):
        return None
    norms = {"mlp": block.post_attention_layernorm, "attn": block.input_layernorm}
    result: dict[str, float] = {}
    for group, norm in norms.items():
        pre: dict[str, torch.Tensor] = {}
        h = norm.register_forward_hook(
            lambda _m, i, _o: pre.__setitem__("x", i[0].detach()[:, -1, :].float())
        )
        got: dict[str, torch.Tensor] = {}
        direct = get_attr(model, f"{prefix}.{layer}.{suffixes[group][0]}")
        h2 = direct.register_forward_hook(
            lambda _m, i, _o: got.__setitem__("x", i[0].detach()[:, -1, :].float())
        )
        try:
            model(input_ids=torch.tensor([list(context) + [7]], device=device))
        finally:
            h.remove()
            h2.remove()
        x = pre["x"]
        analytic = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm.weight)[0]
        result[group] = float((got["x"][0] - analytic).norm() / analytic.norm())
    return result
