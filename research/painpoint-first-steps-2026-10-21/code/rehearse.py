"""Offline rehearsal of the frozen decision rules on synthetic data (CLAUDE.md 6.2).

Every verdict branch the contract relies on must be reachable, and each scenario must
land in the branch its construction implies. Exit code 0 only if all scenarios match.
"""
import sys

import numpy as np

from decide import a1_decide, c1_decide, C1_PRIMARY

D = 20
BASES = ("b0", "b1", "b2", "b3")
N_PER_CLASS = 50


def a1_data(scenario, seed):
    rng = np.random.default_rng(seed)
    X, y, base, split = [], [], [], []
    for bi, b in enumerate(BASES):
        scale = np.ones(D)
        sign = 1.0
        shift = 2.0
        if scenario == "scale":
            # per-base per-feature scale heterogeneity, nuisance features blow up differently
            scale = np.exp(rng.normal(0, 2.5, D))
            scale[:3] = 1.0
        if scenario == "flip" and bi in (1, 3):
            sign = -1.0
        if scenario == "nosignal":
            shift = 0.0
        offset = rng.normal(0, 3, D)                     # per-base offset, shared by both classes
        for lab in (0, 1):
            Z = rng.standard_normal((N_PER_CLASS, D))
            if lab == 1:
                Z[:, :3] += sign * shift
            X.append(Z * scale + offset)
            y += [lab] * N_PER_CLASS
            base += [b] * N_PER_CLASS
            split += ["TRAIN"] * (N_PER_CLASS // 2) + ["TEST"] * (N_PER_CLASS - N_PER_CLASS // 2)
    return np.vstack(X), np.array(y), np.array(base), np.array(split)


def c1_data(scenario, seed):
    rng = np.random.default_rng(seed)
    n_ref = 20 if scenario == "invalid" else 60
    strata = np.array([0] * n_ref + [1] * 50 + [2] * 50 + [3] * 40)
    scores = {}
    for name in C1_PRIMARY:
        s = rng.standard_normal(len(strata))
        if scenario == "shift":
            s[strata > 0] += 1.5
        scores[name] = s
    return scores, strata == 0


def main():
    ok = True
    exp_a = {"same": "A_PREMISE_FALSE", "scale": "A_TRIVIAL_FIX", "flip": "A_OPEN", "nosignal": "STOP_INVALID"}
    for sc, want in exp_a.items():
        X, y, base, split = a1_data(sc, seed=11)
        got, rows = a1_decide(X, y, base, split, seed=1)
        print(f"A1 {sc:9s} want={want:16s} got={got}")
        for r in rows:
            print("    ", r["held"], {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in r.items() if k.startswith(("auc", "tier", "valid"))})
        ok &= got == want
    exp_c = {"iid": "RECIPE_SHIFT_MINOR", "shift": "RECIPE_SHIFT_MAJOR", "invalid": "INVALID"}
    for sc, want in exp_c.items():
        scores, is_ref = c1_data(sc, seed=5)
        got, res = c1_decide(scores, is_ref, seed=2)
        print(f"C1 {sc:9s} want={want:20s} got={got}")
        ok &= got == want
    print("REHEARSAL", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
