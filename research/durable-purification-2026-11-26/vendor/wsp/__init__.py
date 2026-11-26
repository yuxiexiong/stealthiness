"""wsp -- weights-only backdoor trigger recovery.

A reference implementation of the method in:

    S. C. Khilar, "A Backdoor Trigger Is Legible in Fine-Tune Weights."
    https://github.com/nssprogrammer/wsp_backdoor

Please cite the paper if you use this code; see ``CITATION.cff`` or the BibTeX
entry in ``README.md``.

What the method is, in one paragraph
------------------------------------
A backdoored language model behaves normally until a trigger phrase appears in
its input.  Every published method for recovering such a trigger runs the
model.  This one does not: it reads a fine-tuned checkpoint's weight deltas
against a trusted reference, projects them onto a basis measured on the *base*
model, standardises per token against a null built from clean fine-tunes, and
names one of the trigger phrase's own tokens as the single highest-ranked
candidate in the vocabulary.  No forward pass of the suspect, no data, no
labelled population of poisoned models, and no prior knowledge that a backdoor
exists.

Module map
----------
``wsp.arch``       Architecture-generic discovery of the matrices to read.
``wsp.basis``      The context-averaged basis ``x_l[k]``, measured on the base model.
``wsp.salience``   Eq. (1) salience and Eq. (2) per-token standardisation.
``wsp.sites``      The six site-selection rules, including the four that break.
``wsp.recover``    Max-over-sites combination and the three output criteria.
``wsp.designs``    Trigger designs and the hold-out structure they encode.
``wsp.attack``     The adaptive attackers, including the two that win.
``wsp.detect``     Detection scores and clean-calibrated thresholds.

Status of this package
----------------------
The notebooks under ``notebooks/paper/`` are the record of record: every number
in the paper came out of them, and each one is self-contained and reproducible
on a single commodity GPU.  This package is the same logic lifted into
importable modules so it can be read, tested and reused.  Where the two could
drift, trust the notebooks and open an issue.

Scope and intent
----------------
This is defensive security research.  The attacker code in ``wsp.attack`` exists
because a negative result about evasion cannot be checked without the evasions;
it restricts which parameters move or adds a term to a training loss, and
confers no capability for building a backdoor beyond ordinary fine-tuning.
"""

from __future__ import annotations

__version__ = "1.0.0"

__paper__ = "A Backdoor Trigger Is Legible in Fine-Tune Weights"
__author__ = "Snigdha Chandan Khilar"
__url__ = "https://github.com/nssprogrammer/wsp_backdoor"

from . import arch, attack, basis, designs, detect, recover, salience, sites  # noqa: F401

__all__ = [
    "arch",
    "attack",
    "basis",
    "designs",
    "detect",
    "recover",
    "salience",
    "sites",
    "__version__",
]
