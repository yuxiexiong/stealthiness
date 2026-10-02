"""Perturbation sensitivity of the take-off step (CONTRACT_PS.md). numpy only.

All runs: init 1001 and order 1001 (the original s1 sequence), with one minimal change each.
"""
import numpy as np

N, BATCH, MICRO = 20000, 16, 8           # LLaVA: 8 per device x 2 accumulation
STEP = 50                                # perturbations act at step 50 (well before any take-off)
FAR = 250
NOISE, SMALL = 30, 10                    # init-only noise scale (XO) and one 10-step grid cell
KINDS = ("P0", "P1", "P2", "P3", "P4")


def arm(kind):
    return f"PS-{kind}"


def _pos(step, k):
    return BATCH * (step - 1) + k


def perturb(seq, poison, kind):
    """seq: consumed row sequence (randperm(1001)); poison: set of poison rows. Returns a new sequence."""
    s = np.asarray(seq).copy()
    clean = lambda p: int(s[p]) not in poison

    def first_clean(step, lo=0, hi=BATCH):
        for k in range(lo, hi):
            if clean(_pos(step, k)):
                return _pos(step, k)
        raise ValueError("no clean row")

    if kind == "P0":
        return s
    if kind == "P1":                                      # same micro-batch, order inside the sum only
        a = first_clean(STEP, 0, MICRO)
        b = next(p for p in range(a + 1, _pos(STEP, MICRO)) if clean(p))
    elif kind == "P2":                                    # adjacent steps
        a, b = first_clean(STEP), first_clean(STEP + 1)
    elif kind == "P3":                                    # distant steps
        a, b = first_clean(STEP), first_clean(FAR)
    elif kind == "P4":                                    # first poison row at/after STEP moves one step later
        a = next(p for p in range(_pos(STEP, 0), N) if not clean(p))
        st = a // BATCH + 1
        b = first_clean(st + 1)
    else:
        raise ValueError(kind)
    s[a], s[b] = s[b], s[a]
    return s


def verdict(t50):
    """t50: {kind: step or None}. Shifts relative to P0."""
    if any(t50.get(k) is None for k in KINDS):
        return {"tier": "cannot measure (missing runs)"}
    d = {k: abs(t50[k] - t50["P0"]) for k in KINDS[1:]}
    if d["P1"] >= NOISE:
        tier = "numerical-level sensitive (summation order alone moves the take-off)"
    elif d["P2"] >= NOISE or d["P3"] >= NOISE:
        tier = "sample-level sensitive (moving one clean row moves the take-off)"
    elif all(v <= SMALL for v in d.values()):
        tier = "insensitive"
    else:
        tier = "intermediate"
    return {"shift": d, "tier": tier}
