"""Aligned window transplant (CONTRACT_TP.md). numpy only."""
import numpy as np

N, BATCH = 20000, 16
Y = {1007: 470, 1001: 360}            # t50 (10-step grid) at init 1001 for the two whole orders
WINDOWS = [(1, 100), (101, 200), (201, 300)]
HI, LO = 0.75, 0.25


def arm(base, donor, a, b):
    return f"TP-B{base}-D{donor}-W{a}-{b}"


def runs():
    """Three windows of 1001 into 1007, plus 1007's first window into 1001 (control). Init 1001 for all."""
    return [(1007, 1001, a, b) for a, b in WINDOWS] + [(1001, 1007, 1, 100)]


def transplant(base, donor, a, b):
    """Base sequence with the rows the donor consumes at steps a..b placed at the same positions.
    Each displaced base row goes to the base position the incoming row used to occupy (one epoch kept)."""
    base, donor = np.asarray(base), np.asarray(donor)
    P = np.arange(BATCH * (a - 1), BATCH * b)
    S = base.copy()
    rd, rb = donor[P], base[P]
    s_rd, s_rb = set(rd.tolist()), set(rb.tolist())
    S[P] = rd
    pos = np.empty(N, dtype=np.int64)
    pos[base] = np.arange(N)
    dup_pos = sorted(pos[r] for r in rd if r not in s_rb)          # outside P, now duplicated
    missing = [r for r in rb if r not in s_rd]                       # base window rows pushed out
    assert len(dup_pos) == len(missing)
    for q, r in zip(dup_pos, missing):
        S[q] = r
    assert len(np.unique(S)) == N
    return S


def share(base, t):
    """How far the transplant pulled the base toward the donor's whole-order t50."""
    donor = 1001 if base == 1007 else 1007
    return (t - Y[base]) / (Y[donor] - Y[base])


def verdict(obs):
    """obs: {(base, donor, a, b): t50 or None}."""
    if any(v is None for v in obs.values()) or len(obs) < 4:
        return {"tier": "cannot measure (missing runs)"}
    sh = {k: share(k[0], v) for k, v in obs.items()}
    w = [sh[(1007, 1001, a, b)] for a, b in WINDOWS]
    ctrl = sh[(1001, 1007, 1, 100)]
    big = [i for i, s in enumerate(w) if s >= HI]
    if len(big) == 1 and all(s <= LO for i, s in enumerate(w) if i != big[0]):
        a, b = WINDOWS[big[0]]
        tier = f"window {a}-{b} alone carries the early take-off"
    elif all(s <= LO for s in w):
        tier = "no single window transfers the take-off time"
    elif not big and sum(w) >= HI:
        tier = "spread over several windows"
    else:
        tier = "mixed"
    return {"tier": tier, "shares": {f"W{a}-{b}": s for (a, b), s in zip(WINDOWS, w)},
            "control_share": ctrl, "control_label": "delayed" if ctrl >= HI else "unchanged" if ctrl <= LO else "partial"}


def step1_gate(loss, tol=1e-4):
    """W1 transplant into 1007 starts with 1001's first batch (4.7161); the control and the W2/W3
    transplants start with 1007's first batch (4.0838) — the XO gate values."""
    want = {arm(1007, 1001, 1, 100): 4.7161, arm(1001, 1007, 1, 100): 4.0838, arm(1007, 1001, 101, 200): 4.0838}
    d = {k: abs(loss.get(k, float("nan")) - v) for k, v in want.items()}
    return {"diff": d, "pass": all(x <= tol for x in d.values())}


def step1_gate_v2(loss, tol=1e-4):
    """LOG J28: the W101-200 / W201-300 transplants also alter a few first-batch rows (displaced rows land
    anywhere), so only runs whose first batch is provably the source's first batch are checked."""
    want = {arm(1007, 1001, 1, 100): 4.7161, arm(1001, 1007, 1, 100): 4.0838}
    d = {k: abs(loss.get(k, float("nan")) - v) for k, v in want.items()}
    return {"diff": d, "pass": all(x <= tol for x in d.values())}
