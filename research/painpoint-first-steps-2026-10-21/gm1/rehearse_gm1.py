"""Offline rehearsal of the GM1 decision rule on synthetic representations (CLAUDE.md 6.2).

Must show both verdict branches reachable and each scenario landing where its construction implies.
Exit 0 only if all scenarios match.
"""
import sys

import numpy as np

from decide_gm1 import analyse, verdict

D = 256


def synth(n, k, sep_sigma, seed, aniso=3.0):
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.normal(0, np.log(aniso), D))        # anisotropic clean covariance
    H = rng.standard_normal((n, D)) * scale
    poison = np.zeros(n, bool)
    poison[rng.choice(n, k, replace=False)] = True
    direction = rng.standard_normal(D) * scale             # shift lives in a real covariance direction
    direction /= np.linalg.norm(direction / scale)         # unit in whitened metric
    H[poison] += sep_sigma * direction
    return H, poison


def main():
    ok = True
    # strong, coherent, large shift at 1% -> blind viable, kstar small
    H, p = synth(20000, 200, sep_sigma=12.0, seed=1)
    r = analyse(H, p, 20000)
    v = verdict(r)
    print("strong  ", {k: round(v2, 3) if isinstance(v2, float) else v2 for k, v2 in r.items()}, "->", v)
    ok &= v == "GM1_BLIND_VIABLE"
    # weak shift -> blind dead (kstar huge, overlap low)
    H, p = synth(20000, 200, sep_sigma=1.0, seed=2)
    r = analyse(H, p, 20000)
    v = verdict(r)
    print("weak    ", {k: round(v2, 3) if isinstance(v2, float) else v2 for k, v2 in r.items()}, "->", v)
    ok &= v == "GM1_BLIND_DEAD"
    # no shift -> dead, oracle auroc ~0.5
    H, p = synth(20000, 200, sep_sigma=0.0, seed=3)
    r = analyse(H, p, 20000)
    v = verdict(r)
    print("none    ", {k: round(v2, 3) if isinstance(v2, float) else v2 for k, v2 in r.items()}, "->", v)
    ok &= v == "GM1_BLIND_DEAD" and abs(r["oracle_auroc_heldout"] - 0.5) < 0.1
    print("REHEARSAL", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
