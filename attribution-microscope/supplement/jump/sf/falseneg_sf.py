"""False-negative audit of CONTRACT_SF.md's criteria (zero GPU, read-only).

Asks the opposite of rehearsal_sf.py: for how much of the space of GENUINE effects would the
frozen rule print "mixed", i.e. look like no finding? Two parts:

A. Read-out stability: first crossing vs sustained crossing of the median margin, on all 14
   LP runs, to see how much of the assumed 30-step noise is read-out jitter rather than
   training variation.
B. Verdict dead zone: map true skeleton share -> verdict probability, and report which genuine
   worlds are collapsed into "mixed".

    python falseneg_sf.py <results_lp.json>
"""
import json
import sys

import numpy as np

import sf

LP = json.load(open(sys.argv[1]))["runs"]

# ---------------------------------------------------------------- A. read-out stability
print("== A. first crossing vs sustained crossing of the median core margin (all 14 LP runs)")


def sustained(steps, med, hold=40):
    """First crossing after which the median margin stays >= 0 for `hold` steps."""
    pts = sorted(zip(steps, med))
    for i, (s, m) in enumerate(pts):
        if m >= 0 and all(v >= 0 for t, v in pts[i:] if t <= s + hold):
            if i == 0:
                return float(s)
            s0, m0 = pts[i - 1]
            return s0 + (s - s0) * (-m0) / (m - m0) if m0 < 0 else float(s)
    return None


rows, d = [], []
for k, v in LP.items():
    st, med = v["steps"], v["median_M"]
    a, b = sf.t_m0(st, med), sustained(st, med)
    rows.append((k, v["t50"], a, b))
    if a is not None and b is not None:
        d.append(b - a)
print("  run                      t50   first    sustained   diff")
for k, t50, a, b in rows:
    print(f"  {k:24} {t50:>4}  {a:7.1f}  {b:9.1f}  {b - a:+6.1f}" if a and b else f"  {k:24} {t50:>4}  {a}  {b}")
d = np.array(d)
print(f"  diff: median {np.median(d):+.1f}, max {d.max():+.1f}, runs with diff >= 20 steps: {(d >= 20).sum()}/{len(d)}")
print("  PS-P0 (the new reference) is the extreme case: the median margin crosses, dips back")
print("  below zero for ~30 steps, then rises for good.")

# ---------------------------------------------------------------- B. verdict dead zone
print("\n== B. verdict as a function of the TRUE skeleton share (both directions symmetric)")
REF = {1001: 327.9, 1007: 464.5, 1009: 328.1, 1008: 401.5}


def world(share_ab, share_ba, sd, rng):
    obs = {}
    for p, (a, b) in sf.PAIRS.items():
        ta, tb = REF[a], REF[b]
        obs[(p, a, b)] = tb + share_ab * (ta - tb) + rng.normal(0, sd)
        obs[(p, b, a)] = ta + share_ba * (tb - ta) + rng.normal(0, sd)
    return obs


def sweep(sd, n=2000):
    out = []
    for s in np.arange(0.5, 1.01, 0.05):
        c = {}
        for i in range(n):
            r = np.random.default_rng(10000 + i)
            t = sf.verdict(world(s, s, sd, r), REF)["tier"]
            c[t] = c.get(t, 0) + 1
        out.append((s, {k: v / n for k, v in c.items()}))
    return out


for sd in (0.0, 15.0, 30.0):
    print(f"\n  noise sd = {sd:.0f} steps")
    print("   true share | P(skeleton)  P(mixed)   P(other)")
    for s, c in sweep(sd):
        sk = c.get("skeleton sets the take-off", 0.0)
        mx = c.get("mixed", 0.0)
        print(f"      {s:4.2f}    |   {sk:8.3f}  {mx:8.3f}   {1 - sk - mx:8.3f}")

print("\n== B2. asymmetric genuine worlds (one direction strong, the other partial), sd = 0")
for ab, ba in ((1.2, 1.2), (1.2, 0.5), (1.0, 0.5), (0.9, 0.55), (1.0, 0.0), (0.8, 0.8), (0.7, 0.7), (0.65, 0.65)):
    v = sf.verdict(world(ab, ba, 0.0, np.random.default_rng(0)), REF)
    print(f"  shares ({ab:.2f}, {ba:.2f}) -> {v['tier']}")

print("\n== B3. what the XO decomposition actually looked like (real data, for calibration)")
print("  XO order shares were 1.22 / 1.25 / 1.25 / 1.00 and init shares 0.00 / -0.75 / -0.75 / -0.00,")
print("  i.e. the winning factor OVERSHOT the parent interval. A real effect here may well land")
print("  above 1, which the >= 0.6 / mean >= 0.75 rule handles; the risk is the opposite corner,")
print("  a genuine 0.6-0.75 split, which the frozen rule prints as 'mixed'.")
