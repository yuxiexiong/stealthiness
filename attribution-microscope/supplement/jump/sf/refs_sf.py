"""Fetch-and-compute the parent reference take-off times for CONTRACT_SF.md (zero GPU).

    python refs_sf.py <dir with XO-I1007-O1001@s*.json> <dir with PS-P0@s*.json> [out.json]

Both directories hold the LP read-out files produced by logit_probe.py on the server.
The XO directory is a self-check: recomputing t_M0 from the raw read-outs must reproduce
the value in the frozen results_lp.json (358.76) to the digit, otherwise my recomputation
differs from the campaign's and the new reference is not comparable."""
import glob
import json
import os
import re
import sys

import numpy as np

import sf

CHECK_ARM, CHECK_TM0, CHECK_T50 = "XO-I1007-O1001", 358.76056338028167, 360


def curve(d, arm):
    """(steps, median core margin) from the per-checkpoint read-out files."""
    pts = []
    for f in sorted(glob.glob(os.path.join(d, f"{arm}@s*.json"))):
        s = int(re.search(r"@s(\d+)\.json$", f).group(1))
        rec = json.load(open(f))
        pts.append((s, float(np.median([r["M"] for r in rec["core"]])), rec))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def agree(recs):
    """lp.gate's quantity on what these files contain: argmax_is_target vs the probe's own
    greedy ASR. p_core has no greedy answer, so this is computable on p_seen only."""
    pairs = [(r["argmax_is_target"], r["asr"]) for rec in recs for r in rec["seen"]]
    return (float(np.mean([a == b for a, b in pairs])), len(pairs)) if pairs else (float("nan"), 0)


XO_DIR, PS_DIR = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None

st, med, recs = curve(XO_DIR, CHECK_ARM)
tm_check = sf.t_m0(st, med)
a, n = agree(recs)
ok = tm_check is not None and abs(tm_check - CHECK_TM0) < 1e-6
print(f"[self-check] {CHECK_ARM}: {len(st)} checkpoints, recomputed t_M0 = {tm_check}")
print(f"[self-check] frozen results_lp.json says {CHECK_TM0} -> {'MATCH' if ok else 'MISMATCH'}")
print(f"[self-check] p_seen argmax vs greedy agreement {a:.4f} on {n} pairs")
if not ok:
    print("ABORT: my recomputation does not reproduce the campaign value; the new reference "
          "would not be on the same footing. Do not freeze.")
    sys.exit(1)

st, med, recs = curve(PS_DIR, "PS-P0")
tm = sf.t_m0(st, med)
a2, n2 = agree(recs)
print(f"\n[reference] PS-P0 (init 1001, order 1001 = original s1): {len(st)} checkpoints")
print(f"[reference] t_M0 = {tm:.2f}" if tm is not None else "[reference] t_M0 = None (never crosses)")
print(f"[reference] p_seen argmax vs greedy agreement {a2:.4f} on {n2} pairs")
w = [(s, m) for s, m in zip(st, med) if 280 <= s <= 400]
print("[reference] median core margin near the crossing: " + ", ".join(f"{s}:{m:+.2f}" for s, m in w))

refs = {1007: 464.48275862068965, 1008: 401.5, 1009: 328.1, 1001: tm}
print("\n[gap rule, CONTRACT §2.2] judged pairs must have a t_M0 gap >= 119 steps")
for p, (x, y) in sf.PAIRS.items():
    g = abs(refs[x] - refs[y])
    role = "judged" if p in sf.JUDGED else "descriptive"
    verdict = "OK" if g >= 119 else "BELOW 119 -> demote to descriptive"
    print(f"  {p} ({x} vs {y}, {role}): gap {g:.1f}  n* = {15.7 * (sf.NOISE_SD / g) ** 2:.2f}  {verdict}")

if OUT:
    json.dump({str(k): v for k, v in refs.items()}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
