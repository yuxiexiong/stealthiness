"""The six site-selection rules, including the four that are broken.

Reference
---------
S. C. Khilar, "A Backdoor Trigger Is Legible in Fine-Tune Weights."
Section VIII (What Survives Is the Rule That Reads Least), with the
head-to-head in Table I and the architecture replication in Table II.
See ``docs/METHOD.md``.

What this implements
--------------------
Having a statistic is not enough: an auditor must decide *which* sites to read
before combining them.  The paper's second contribution is negative, and this
module is where it lives.  All six rules are kept, not just the surviving one,
because the broken ones are the result.  Deleting them would delete the
evidence.

======  ======================================  ==========================================
Rule    What it conditions on                   How it dies
======  ======================================  ==========================================
R1      Exact-zero deltas                       Camouflage: retraining the frozen block on
                                                clean data makes the deltas nonzero again.
R2      Cross-model norm ratio vs clean median  Three ways: norm injection, a 0.25x
                                                learning rate, and longer camouflage
                                                (GPT-2 reaches rho = 0.954 against
                                                theta = 0.682).
R4      Within-model ratio vs median over ALL   Degenerate by construction: with more than
        depths                                  half the scanned depths frozen the median
                                                is 0, so every live site passes.
R4b     Within-model ratio over NONZERO sites   Survives longer, but its own slope predicts
                                                failure at 16--32 camouflage epochs.
R5      Fixed window, layers 0--8               Reads nothing from the suspect, so nothing
                                                can be steered -- but a frozen window
                                                leaves it with an all-zero delta and no
                                                valid null.
R6      R5, plus drop exactly-zero sites, plus  **The paper's rule.** Best-or-tied against
        extend when the whole window is zero    every attacker on SmolLM2 and GPT-2.  Still
                                                defeated by the composite and the gradient
                                                attackers (Section VII).
======  ======================================  ==========================================

The lesson the paper draws, and the reason R6 is written the way it is: *any*
selection rule conditioned on the suspect's weight magnitudes hands the
attacker a dial.  R6 conditions on the suspect only through a test for
exactly-zero deltas, which is the least an auditor can read and still have a
valid null.

Feeds
-----
Tables I and II directly; every recovery number indirectly, since a rule must
be chosen before the statistic can be combined.

Lifted from
-----------
``notebooks/paper/backdoor_v18_full.ipynb``, Cell 6 (all six rules) and
``notebooks/paper/backdoor_v20_cost.ipynb``, Cell 6 (five rules, GPT-2).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = [
    "Site",
    "DEFAULT_WINDOW",
    "ZERO_TOL",
    "sites_R1",
    "sites_R2",
    "sites_R4",
    "sites_R4b",
    "sites_R5",
    "sites_R6",
    "RULES",
    "clean_threshold",
]

#: A site is ``(layer, group)`` with group in ``{"mlp", "attn"}``.
Site = tuple[int, str]

#: Top of the fixed scan window used by R5 and R6.  The published runs scan
#: layers 0, 2, 4, 6, 8 within it.  Section VIII explains why the window is
#: fixed in advance rather than chosen from the suspect.
DEFAULT_WINDOW = 8

#: Below this Frobenius norm a site's delta counts as exactly zero.  Frozen
#: parameters produce an exact 0.0; the tolerance absorbs float round-off in
#: checkpoint round-tripping, nothing more.
ZERO_TOL = 1e-10

_GROUPS = ("mlp", "attn")


def _first_k(items: Sequence[int], k: int) -> list[int]:
    """Shallowest ``k`` depths.  The paper's rules all read the earliest sites."""
    return list(items[:k])


def sites_R1(norms: dict[Site, float], scan: Sequence[int], k: int = 3, **_: object) -> list[Site]:
    """R1 -- the shallowest ``k`` sites whose delta is not exactly zero.

    Broken by camouflage (Section VII): one clean epoch over the frozen block
    makes every delta nonzero, so R1 happily selects sites the attacker has
    scrubbed.
    """
    return [
        (layer, group)
        for group in _GROUPS
        for layer in _first_k([l for l in scan if norms[(l, group)] > ZERO_TOL], k)
    ]


def sites_R2(
    norms: dict[Site, float],
    scan: Sequence[int],
    clean_median: dict[Site, float],
    theta: float,
    k: int = 3,
    **_: object,
) -> list[Site]:
    """R2 -- sites whose norm reaches ``theta`` times the clean median there.

    Broken three ways (Section VII): norm injection fills a frozen site to
    clean scale; a 0.25x learning rate shrinks every real site below theta;
    and longer camouflage walks the ratio back up past it.
    """
    return [
        (layer, group)
        for group in _GROUPS
        for layer in _first_k(
            [
                l
                for l in scan
                if clean_median[(l, group)] > 0
                and norms[(l, group)] / clean_median[(l, group)] >= theta
            ],
            k,
        )
    ]


def _within_all(norms: dict[Site, float], scan: Sequence[int], layer: int, group: str) -> float:
    """R4's ratio: this site's norm over the median across *all* scanned depths."""
    med = float(np.median([norms[(l, group)] for l in scan]))
    return norms[(layer, group)] / med if med > 0 else float("inf")


def _within_nonzero(norms: dict[Site, float], scan: Sequence[int], layer: int, group: str) -> float:
    """R4b's ratio: the same, but the median is taken over nonzero sites only."""
    live = [norms[(l, group)] for l in scan if norms[(l, group)] > ZERO_TOL]
    med = float(np.median(live)) if live else 0.0
    return norms[(layer, group)] / med if med > 0 else float("inf")


def sites_R4(
    norms: dict[Site, float], scan: Sequence[int], theta_within: float, k: int = 3, **_: object
) -> list[Site]:
    """R4 -- within-model ratio against the median over all depths.

    Kept because it is instructive, not because it works.  With more than half
    the scanned depths frozen the median is exactly 0, the ratio is infinite
    everywhere, and the rule degenerates to "select everything live".
    """
    return [
        (layer, group)
        for group in _GROUPS
        for layer in _first_k(
            [l for l in scan if _within_all(norms, scan, l, group) >= theta_within], k
        )
    ]


def sites_R4b(
    norms: dict[Site, float], scan: Sequence[int], theta_within: float, k: int = 3, **_: object
) -> list[Site]:
    """R4b -- R4 with the median taken over nonzero sites, which repairs R4's degeneracy.

    Survives the attackers R4 does not, but is still a magnitude rule: its own
    measured slope under the camouflage ladder (Table II) predicts failure
    somewhere between 16 and 32 camouflage epochs.
    """
    return [
        (layer, group)
        for group in _GROUPS
        for layer in _first_k(
            [l for l in scan if _within_nonzero(norms, scan, l, group) >= theta_within], k
        )
    ]


def sites_R5(scan: Sequence[int], window: int = DEFAULT_WINDOW, **_: object) -> list[Site]:
    """R5 -- a fixed window, layers 0..``window``, reading nothing from the suspect.

    Nothing about the suspect steers it, which is the property the paper wants.
    Its failure mode is narrow but fatal: against a full freeze of the window
    every selected delta is exactly zero, the null has no spread, and the
    statistic is undefined.
    """
    return [(layer, group) for layer in scan if layer <= window for group in _GROUPS]


def sites_R6(
    norms: dict[Site, float],
    scan: Sequence[int],
    window: int = DEFAULT_WINDOW,
    k: int = 3,
    **_: object,
) -> list[Site]:
    """R6 -- the paper's rule: fixed window, drop exact zeros, extend if all zero.

    Three clauses, in order:

    1. Take the fixed window (R5).  Nothing in it depends on the suspect.
    2. Drop sites whose delta is exactly zero.  A frozen site carries no
       information and no valid null; an *injected* site is inert rather than
       misleading, so a nonzero injected site is deliberately kept.
    3. If the whole window is zero, follow the freeze outward to the shallowest
       live sites.  Without this clause the rule returns nothing at all against
       a full-window freeze.

    This conditions on the suspect only through a test for exactly-zero deltas.
    It is best-or-tied against every attacker in the paper on both
    architectures, and it is still defeated by the composite attacker (0/16,
    for free) and the gradient attacker (0/2 at every penalty strength).
    """
    win = [(layer, group) for layer in scan if layer <= window for group in _GROUPS]
    live = [s for s in win if norms[s] > ZERO_TOL]
    if live:
        return live
    out: list[Site] = []
    for group in _GROUPS:
        out += [
            (layer, group)
            for layer in _first_k([l for l in scan if norms[(l, group)] > ZERO_TOL], k)
        ]
    return out


#: All six rules, in the order Table I reports them.  The broken ones are part
#: of the result; do not prune this mapping.
RULES: dict[str, Callable[..., list[Site]]] = {
    "R1 exact-zero": sites_R1,
    "R2 cross-model": sites_R2,
    "R4 broken": sites_R4,
    "R4b nonzero-med": sites_R4b,
    "R5 fixed 0-8": sites_R5,
    "R6 fixed+zerodrop": sites_R6,
}


def clean_threshold(clean_ratios: Sequence[float], margin: float = 0.8) -> float:
    """Calibrate a threshold from clean models alone, as R2 and R4b require.

    The published runs set ``theta = 0.8 * min(clean ratio)``.  Calibrating on
    the clean population is what keeps the threshold auditor-side; the point of
    Section VIII is that it does not help, because the attacker can move the
    suspect's ratio wherever they like.
    """
    if not len(clean_ratios):
        raise ValueError("need at least one clean ratio to calibrate a threshold")
    return margin * float(min(clean_ratios))
