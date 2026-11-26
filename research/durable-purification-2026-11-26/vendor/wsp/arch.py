"""Architecture-generic discovery of the matrices the statistic reads.

Reference
---------
S. C. Khilar, "A Backdoor Trigger Is Legible in Fine-Tune Weights."
Section V (Method), paragraph "Sites".  See ``docs/METHOD.md``.

What this implements
--------------------
A *site* is a layer's input-matrix group: the MLP group ``{W_gate, W_up}`` or
the attention group ``{W_Q, W_K, W_V}``.  The paper's claim is that the method
is architecture-generic, so nothing here may hard-code a module path.  We walk
the module tree of layer 0, keep every 2-D weight whose leaf name is a known
input projection, and classify it.  That is what lets one pipeline run over
Llama-style (SmolLM2, Qwen2), GPT-2 and GPT-NeoX checkpoints unchanged
(Table IX).

Two architecture facts bite, and both are handled here rather than at the call
site:

* GPT-2 stores its projections as ``Conv1D``, i.e. transposed relative to
  ``nn.Linear``.  ``oriented()`` restores ``(d_out, d_in)`` so the Gram
  factorisation in :mod:`wsp.salience` is orientation-free.
* Fused projections (``c_attn``, ``query_key_value``, ``Wqkv``) carry Q, K and
  V in one matrix.  They are treated as a single member of the attention
  group, which is what the group-mean in the statistic already assumes.

Feeds
-----
Table IX (recovery across architecture families), and every table produced by
``backdoor_v19_arch.ipynb`` / ``backdoor_v20_cost.ipynb``.

Lifted from
-----------
``notebooks/paper/backdoor_v20_cost.ipynb``, Cell 1.  That notebook is the
record of record; this module is the same code made importable.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MLP_IN",
    "ATT_IN",
    "LAYER_PATHS",
    "TRANSPOSED",
    "get_attr",
    "discover",
    "oriented",
    "matrix_key",
]

# Leaf names of the input-side projections, across the families in Table IX.
# MLP group: the matrices that read the residual stream before the non-linearity.
MLP_IN = [
    "gate_proj", "up_proj",        # Llama / Qwen2 / SmolLM2
    "c_fc",                        # GPT-2
    "dense_h_to_4h",               # GPT-NeoX / Pythia
    "fc_in",                       # GPT-J
    "w1", "w3",                    # Mistral-style naming
    "gate_up_proj",                # fused gate+up
]

# Attention group: the matrices that read the residual stream to form Q, K, V.
ATT_IN = [
    "q_proj", "k_proj", "v_proj",  # Llama / Qwen2 / SmolLM2
    "c_attn",                      # GPT-2 (fused QKV)
    "query_key_value",             # GPT-NeoX (fused QKV)
    "Wqkv", "qkv_proj",            # other fused spellings
]

# Where the list of transformer blocks lives, by family.
LAYER_PATHS = [
    "model.layers",                # Llama / Qwen2 / SmolLM2
    "transformer.h",               # GPT-2
    "gpt_neox.layers",             # Pythia
    "model.decoder.layers",        # OPT
    "transformer.blocks",          # MPT
]

# Module classes that store weights transposed relative to nn.Linear.
TRANSPOSED = ("Conv1D",)


def get_attr(obj: Any, path: str) -> Any:
    """Follow a dotted attribute path, returning ``None`` if any hop is absent."""
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def discover(model: Any) -> tuple[str, dict[str, list[str]], int]:
    """Locate the transformer blocks and classify their input matrices.

    Parameters
    ----------
    model:
        A loaded causal-LM.  Only its module tree is inspected; no forward
        pass is run, which is the point of the whole method (Section IV).

    Returns
    -------
    (prefix, suffixes, n_layers)
        ``prefix`` is the dotted path to the layer container, ``suffixes`` maps
        ``"mlp"`` and ``"attn"`` to the within-layer paths of their input
        matrices, and ``n_layers`` is the block count.

    Raises
    ------
    RuntimeError
        If no layer container is found, or if either group comes back empty.
        Failing loudly matters: a silently empty group would make the statistic
        read nothing and report a confident miss.
    """
    prefix = next((p for p in LAYER_PATHS if get_attr(model, p) is not None), None)
    if prefix is None:
        raise RuntimeError(
            "no layer container found; add this architecture's path to LAYER_PATHS"
        )
    layers = get_attr(model, prefix)
    names = [
        name
        for name, mod in layers[0].named_modules()
        if getattr(mod, "weight", None) is not None and mod.weight.dim() == 2
    ]
    mlp = [n for n in names if n.split(".")[-1] in MLP_IN]
    att = [n for n in names if n.split(".")[-1] in ATT_IN]
    if not mlp or not att:
        raise RuntimeError(
            f"could not classify input matrices; layer 0 exposes {names}. "
            "Add the missing leaf names to MLP_IN / ATT_IN."
        )
    return prefix, {"mlp": mlp, "attn": att}, len(layers)


def oriented(weight: Any, d_in: int, module: Any | None = None) -> Any:
    """Return ``weight`` as ``(d_out, d_in)`` whatever the storage convention.

    GPT-2's ``Conv1D`` stores ``(d_in, d_out)``.  Shape alone is ambiguous when
    ``d_in == d_out``, so the module class decides when it is available and the
    shape is only a fallback.
    """
    if module is not None and type(module).__name__ in TRANSPOSED:
        return weight.t()
    if weight.shape[1] == d_in:
        return weight
    if weight.shape[0] == d_in:
        return weight.t()
    raise RuntimeError(
        f"matrix {tuple(weight.shape)} fits neither orientation for d_in={d_in}"
    )


def matrix_key(prefix: str, layer: int, suffix: str) -> str:
    """State-dict key of one input matrix, e.g. ``model.layers.6.mlp.up_proj.weight``."""
    return f"{prefix}.{layer}.{suffix}.weight"
