"""Input-side salience, and its per-token standardisation.

Reference
---------
S. C. Khilar, "A Backdoor Trigger Is Legible in Fine-Tune Weights."
Section V (Method), Eq. (1) for the salience and Eq. (2) for the
standardisation.  See ``docs/METHOD.md``.

What this implements
--------------------
**Eq. (1), salience.**  For a scanned site (a layer ``l`` and a group ``M`` of
input matrices) and a vocabulary token ``k``::

    sigma_l(k) = (1 / |M_l|) * sum over M in M_l of || dW^M_l @ x_l[k] ||

where ``dW`` is the suspect-minus-reference weight delta and ``x_l[k]`` is the
context-averaged basis from :mod:`wsp.basis`.  No forward pass of the suspect
is involved anywhere in this file; that is the paper's central claim.

The norm is computed through the Gram factorisation::

    || dW @ x ||^2 = x^T (dW^T dW) x

which turns a per-token matrix-vector product into one ``(d, d)`` Gram matrix
reused across the whole vocabulary.  This is what makes a full-vocabulary scan
affordable on a single commodity GPU.  ``clamp_min(0)`` before the square root
guards against small negative values from float round-off, not against a real
negative quantity.

**Eq. (2), standardisation.**  Raw salience is dominated by per-token nuisance:
two unrelated clean fine-tunes correlate at r = +0.78 (Section IX).  Each token
is therefore standardised against a null built from clean--clean pairs::

    z(k) = (sigma(k) - mu_null(k)) / s_null(k)

with ``s_null`` floored at its 20th percentile.  The floor matters: without it
a token whose null happens to be nearly constant acquires an enormous z from
ordinary noise.  Table VIII reports what removing the standardisation costs --
recovery falls to zero on the designs where it was measured both ways.

**Scale normalisation.**  :func:`scale_factors` rescales each model's delta at
a site to the clean median norm there, so the statistic stops depending on the
suspect's learning rate.  This is the fix for the recipe-mismatch false
positives in Section X; ``1.0`` recovers the published statistic exactly.

Feeds
-----
Every table in the paper.

Lifted from
-----------
``notebooks/paper/backdoor_v18_full.ipynb``, Cell 4, and the architecture-generic
form in ``notebooks/paper/backdoor_v20_cost.ipynb``, Cell 4.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import torch

__all__ = [
    "site_basis",
    "salience",
    "standardise",
    "delta_norm",
    "scale_factors",
    "NUISANCE_CORRELATION",
]

#: Correlation between the raw per-token salience of two unrelated clean
#: fine-tunes (Section IX).  Reported here because it is the reason Eq. (2)
#: exists: without standardisation this nuisance dominates the signal.
NUISANCE_CORRELATION = 0.78


def site_basis(
    basis: dict[tuple[str, int], torch.Tensor],
    layer: int,
    group: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Fetch ``x_l[k]`` for one site as a float32 tensor on ``device``."""
    return basis[(group, layer)].to(device, torch.float32)


@torch.no_grad()
def salience(
    basis_matrix: torch.Tensor,
    deltas: Iterable[torch.Tensor],
    device: str = "cuda",
) -> np.ndarray:
    """Eq. (1): mean over a site's matrices of ``|| dW x_k ||``, for every token.

    Parameters
    ----------
    basis_matrix:
        ``(vocab, d_in)`` context-averaged basis for this site.
    deltas:
        The suspect-minus-reference deltas of this site's matrices, each
        ``(d_out, d_in)``.  Orientation is the caller's responsibility; use
        :func:`wsp.arch.oriented`.

    Returns
    -------
    np.ndarray
        ``(vocab,)`` salience, on CPU.

    Notes
    -----
    The Gram matrix is freed per matrix rather than accumulated, which keeps
    peak memory at one ``(d_in, d_in)`` block.  On a 16 GB T4 this is what
    allows a 49,152-token vocabulary to be scanned without offloading.
    """
    deltas = list(deltas)
    if not deltas:
        raise ValueError("a site must contain at least one matrix")
    total = torch.zeros(basis_matrix.shape[0], device=device)
    for dw in deltas:
        dw = dw.to(device, torch.float32)
        gram = dw.T @ dw
        total += ((basis_matrix @ gram) * basis_matrix).sum(1).clamp_min(0).sqrt()
        del dw, gram
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return (total / len(deltas)).cpu().numpy()


def standardise(
    raw: np.ndarray,
    null: np.ndarray,
    sd_floor_percentile: float = 20.0,
) -> np.ndarray:
    """Eq. (2): per-token z-score against a clean--clean null.

    Parameters
    ----------
    raw:
        ``(vocab,)`` salience of the suspect against the reference.
    null:
        ``(n_pairs, vocab)`` salience over clean--clean pairs.  The published
        runs use the ``C(n, 2)`` pairs available from the clean population.
    sd_floor_percentile:
        The null standard deviation is floored at this percentile of itself.
        20 is the published value; raising it is a strictly more conservative
        choice, lowering it inflates z for low-variance tokens.

    Returns
    -------
    np.ndarray
        ``(vocab,)`` standardised statistic.
    """
    if null.ndim != 2:
        raise ValueError("null must be (n_pairs, vocab)")
    if null.shape[0] < 2:
        raise ValueError("the null needs at least two clean--clean pairs to have a spread")
    sd = null.std(0)
    sd = np.maximum(sd, np.percentile(sd, sd_floor_percentile)) + 1e-9
    return (raw - null.mean(0)) / sd


def delta_norm(deltas: Iterable[torch.Tensor]) -> float:
    """Frobenius norm of a whole site's delta, used by the site rules."""
    return float(torch.sqrt(sum(d.pow(2).sum() for d in deltas)).item())


def scale_factors(
    norms: dict[str, dict[tuple[int, str], float]],
    clean_median: dict[tuple[int, str], float],
    model: str,
    site: tuple[int, str],
) -> float:
    """Rescale a model's delta at one site to the clean median norm there.

    Returns ``1.0`` -- the published statistic -- when either norm is zero, so
    a frozen or injected site is never silently amplified.  Section X reports
    that this is what removes the recipe-mismatch false positives (0/6 against
    2/6) while still flagging all 50 backdoors.
    """
    med = clean_median.get(site, 0.0)
    own = norms[model].get(site, 0.0)
    if med > 0 and own > 0:
        return med / own
    return 1.0
