"""Detection: deciding whether a checkpoint is backdoored at all.

Reference
---------
S. C. Khilar, "A Backdoor Trigger Is Legible in Fine-Tune Weights."
Section X (Detection).

What this implements
--------------------
Recovery answers "what is the trigger"; detection answers the prior question,
"is there one".  The score is the peak of the standardised statistic -- how far
the single most salient token stands above its own clean--clean null.  A
backdoored model has one token pulled far out of the distribution; a clean
fine-tune does not.

**Scale normalisation is the operative detail.**  The raw score depends on how
hard the suspect was trained, so a clean model fine-tuned at twice the
defender's assumed learning rate scores like a backdoored one.  That is a false
positive produced by a recipe mismatch, not by a backdoor.  Rescaling each
site's delta to the clean median norm before scoring removes it: 0 of 6 false
positives against 2 of 6 raw, while all 50 backdoors are still flagged.

**What this does not do.**  It reports a score and a threshold calibrated on
clean models; it does not claim a calibrated false-positive rate for a
population the auditor has not sampled.  Section XI is explicit that a defence
built on this statistic would have to report a small set of candidates and
wear the false-positive rate that comes with it.

An honest caveat recorded in the paper and repeated here: sites filled by norm
injection are **not** inert.  Measured detection scores put grafted sites at
roughly twice a clean draw.  An earlier version of this work asserted the
opposite; the measurement refuted it.

Feeds
-----
Section X; the detection rows of Tables I and II.

Lifted from
-----------
``notebooks/paper/backdoor_v18_full.ipynb`` Cell 7 and
``notebooks/paper/backdoor_v20_cost.ipynb`` Cell 8.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["detection_score", "calibrate_threshold", "flag"]


def detection_score(combined: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Peak standardised salience: the score for one checkpoint.

    Parameters
    ----------
    combined:
        The max-over-sites standardised statistic from
        :func:`wsp.recover.combine_sites`.
    mask:
        Optional eligibility mask.  The published detection numbers score the
        peak over *eligible* tokens, matching what an auditor would act on.
    """
    scores = np.where(mask, combined, -np.inf) if mask is not None else combined
    return float(np.max(scores))


def calibrate_threshold(clean_scores: Sequence[float], margin: float = 1.0) -> float:
    """Threshold from the clean population alone: ``max(clean) * margin``.

    Calibrating on clean models is what keeps the threshold auditor-side.  It
    is also why a recipe mismatch breaks it, and why scale normalisation is
    applied before scoring rather than after.
    """
    if not len(clean_scores):
        raise ValueError("need a clean population to calibrate a detection threshold")
    return float(max(clean_scores)) * margin


def flag(score: float, threshold: float) -> bool:
    """Whether a checkpoint is flagged.  Strictly above, so ties stay unflagged."""
    return score > threshold
