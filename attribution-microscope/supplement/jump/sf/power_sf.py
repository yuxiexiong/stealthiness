"""Pre-freeze power accounting for CONTRACT_SF.md (zero GPU, read-only).

Reports, for each candidate pair: the parent reference take-off times, their gap, the
noise-to-gap ratio in share units, and the shuffle-seed budget of arXiv:2606.29554 Eq. 18,
n* >= 2 sigma^2 (z_{1-a/2} + z_{1-b})^2 / Delta^2 ~= 15.7 (sigma/Delta)^2 at alpha 0.05, power 0.80.

    python power_sf.py <results_lp.json>
"""
import json
import sys

import numpy as np

import sf

LP = sys.argv[1] if len(sys.argv) > 1 else "../lp/results_lp.json"
runs = json.load(open(LP))["runs"]
tm = {k: v["t_M0"] for k, v in runs.items()}

print("== t_M0 (margin crossing, lp.py) for runs with init 1001; the order is the seed in the name")
for k in ("XO-I1001-O1007", "XO-I1001-O1008", "XO-I1001-O1009", "XO-I1001-O1011"):
    print(f"   {k:16} order {k[-4:]}  t_M0 {tm[k]:7.1f}")
print("   order 1001 at init 1001 (= original s1 / PS-P0): NOT in the repo;")
print("   its LP read-out is on the server and must be fetched before freezing (CONTRACT §2).")
print("   in-repo runs that USE order 1001 (other inits), for scale only:")
for k in ("XO-I1007-O1001", "XO-I1009-O1001"):
    print(f"   {k:16} t_M0 {tm[k]:7.1f}")

PLACEHOLDER = 358.2   # mean of the two order-1001 runs above; NOT used in any verdict
ref = {1007: tm["XO-I1001-O1007"], 1008: tm["XO-I1001-O1008"],
       1009: tm["XO-I1001-O1009"], 1001: PLACEHOLDER}
SD = sf.NOISE_SD
Z = 15.7

print(f"\n== pair accounting on t_M0 (order 1001 = {PLACEHOLDER} PLACEHOLDER), sigma = {SD:.0f} steps")
print("  pair  A     B      ref_A   ref_B    gap   sigma/gap   n* per side")
for p, (a, b) in sf.PAIRS.items():
    g = abs(ref[a] - ref[b])
    print(f"  {p}   {a}  {b}  {ref[a]:7.1f} {ref[b]:7.1f} {g:6.1f}  {SD / g:9.2f}  {Z * (SD / g) ** 2:10.2f}")

print("\n== same accounting on the 10-step ASR grid (t50, J38-corrected references)")
for p, (a, b) in sf.PAIRS.items():
    g = abs(sf.REF_T50[a] - sf.REF_T50[b])
    n = Z * (SD / g) ** 2 if g else float("inf")
    print(f"  {p}   gap {g:3d}   sigma/gap {SD / g:.2f}   n* {n:.2f}")

print(f"\n== smallest gap one order per side resolves at sigma = {SD:.0f} (n* <= 1): "
      f"{SD * np.sqrt(Z):.0f} steps")
print("== measured noise floors in this setup")
print("   same init + same order (deterministic trainer; PS-P0 vs original s1): loss and ASR identical -> 0")
print("   micro-perturbations of one fixed order (PS P1-P4): t50 shift 0 / 0 / 0 / 0 -> 0")
t = [330, 330, 360, 360]
print(f"   order 1001, four different inits (XO): t50 {t} -> range {max(t) - min(t)}, sd {np.std(t, ddof=1):.1f}")
print("   => sigma = 30 is the init-change band, used here as a conservative stand-in for")
print("      'how far an irrelevant structural change can move the read-out'. Within a fixed")
print("      (init, order) there is no repeat-measurement noise at all.")
