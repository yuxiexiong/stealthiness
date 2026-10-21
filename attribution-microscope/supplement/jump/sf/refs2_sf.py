"""Consequence of the false-negative audit: recompute every parent reference on the SUSTAINED
crossing as well, and re-apply the 119-step gap rule. Zero GPU.

    python refs2_sf.py <results_lp.json> <dir with PS-P0@s*.json>
"""
import glob
import json
import re
import sys

import numpy as np

import sf
from falseneg_sf import sustained

LP = json.load(open(sys.argv[1]))["runs"]
PS_DIR = sys.argv[2]

pts = []
for f in sorted(glob.glob(PS_DIR + "/PS-P0@s*.json")):
    s = int(re.search(r"@s(\d+)\.json$", f).group(1))
    pts.append((s, float(np.median([r["M"] for r in json.load(open(f))["core"]]))))
pts.sort()
st, med = [p[0] for p in pts], [p[1] for p in pts]
first_1001, sust_1001 = sf.t_m0(st, med), sustained(st, med)
print(f"PS-P0 (init 1001, order 1001): first crossing {first_1001:.2f}, sustained {sust_1001:.2f}")

ref_first = {1001: first_1001}
ref_sust = {1001: sust_1001}
for seed, arm in ((1007, "XO-I1001-O1007"), (1008, "XO-I1001-O1008"), (1009, "XO-I1001-O1009")):
    v = LP[arm]
    ref_first[seed] = sf.t_m0(v["steps"], v["median_M"])
    ref_sust[seed] = sustained(v["steps"], v["median_M"])
print("\nparent references (init 1001 throughout)")
print("  order | first crossing | sustained | t50 (10-step grid)")
for s in (1001, 1007, 1008, 1009):
    print(f"  {s}  |     {ref_first[s]:7.1f}    |  {ref_sust[s]:7.1f}  |   {sf.REF_T50[s]}")

print("\n119-step gap rule applied to each candidate read-out")
print("  pair        first-crossing gap   n*     sustained gap   n*     t50 gap   n*")
for p, (a, b) in sf.PAIRS.items():
    gf, gs = abs(ref_first[a] - ref_first[b]), abs(ref_sust[a] - ref_sust[b])
    gt = abs(sf.REF_T50[a] - sf.REF_T50[b])
    n = lambda g: 15.7 * (sf.NOISE_SD / g) ** 2
    print(f"  {p} ({a}/{b})   {gf:8.1f}  {n(gf):6.2f}      {gs:8.1f}  {n(gs):6.2f}    {gt:5.0f}  {n(gt):6.2f}")

print("\nreading: on the SUSTAINED read-out the 1001-based pairs shrink, because 1001's")
print("sustained crossing is ~27 steps later than its first crossing (its median margin dips")
print("back below zero once). P2 (1009 vs 1007) is the only pair that clears 119 steps on all")
print("three read-outs, so it is the one pair whose verdict does not depend on which crossing")
print("rule is primary.")
json.dump({"first": {str(k): v for k, v in ref_first.items()},
           "sustained": {str(k): v for k, v in ref_sust.items()},
           "t50": {str(k): v for k, v in sf.REF_T50.items()}},
          open("refs_tm0.json", "w"), indent=1)
print("\nrewrote refs_tm0.json with all three read-outs")
