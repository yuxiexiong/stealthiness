"""Offline rehearsal for CONTRACT_PS.md."""
import numpy as np

import ps

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


rng = np.random.default_rng(0)
seq = rng.permutation(ps.N)
poison = set(rng.choice(ps.N, 200, replace=False).tolist())
for k in ps.KINDS:
    s = ps.perturb(seq, poison, k)
    diff = np.flatnonzero(s != seq)
    steps = sorted({int(p) // ps.BATCH + 1 for p in diff})
    one_epoch = len(np.unique(s)) == ps.N
    first_ok = (s[:16] == seq[:16]).all()
    if k == "P0":
        check("P0 identical", len(diff) == 0)
    elif k == "P1":
        check(f"P1: two rows inside one micro-batch of step 50 {diff.tolist()}",
              len(diff) == 2 and steps == [50] and diff[0] // 8 == diff[1] // 8)
    elif k == "P2":
        check(f"P2: steps {steps}", len(diff) == 2 and steps == [50, 51])
    elif k == "P3":
        check(f"P3: steps {steps}", len(diff) == 2 and steps == [50, 250])
    else:
        moved = [int(seq[p]) for p in diff]
        check(f"P4: steps {steps}, one poison row moved one step later",
              len(diff) == 2 and sum(r in poison for r in moved) == 1 and steps[1] - steps[0] == 1)
    check(f"{k}: one epoch, first batch unchanged", one_epoch and first_ok)
check("verdict numerical", ps.verdict({"P0": 360, "P1": 400, "P2": 360, "P3": 360, "P4": 360})["tier"].startswith("numerical"))
check("verdict sample", ps.verdict({"P0": 360, "P1": 370, "P2": 400, "P3": 360, "P4": 360})["tier"].startswith("sample"))
check("verdict insensitive", ps.verdict({"P0": 360, "P1": 360, "P2": 370, "P3": 350, "P4": 360})["tier"] == "insensitive")
check("verdict intermediate", ps.verdict({"P0": 360, "P1": 360, "P2": 380, "P3": 360, "P4": 360})["tier"] == "intermediate")
check("verdict missing", ps.verdict({"P0": 360, "P1": None, "P2": 380, "P3": 360, "P4": 360})["tier"].startswith("cannot"))
print(f"{sum(ok)}/{len(ok)}")
