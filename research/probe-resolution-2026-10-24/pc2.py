"""PC-2: how much of the band is probe-sampling noise?  Pre-declared before running.

Contract addendum (frozen before any number was read):
  GATE  reproduce the published t_M0s from the files with the FROZEN sf module, to within
        0.01 step, for CL-ID (323.0769230769231) and XO-I1001-O1009 (328.07).  Fail -> stop.
  STAT  resample the 200 probes with replacement (B=2000), recompute the median margin
        curve and sf.t_m0_sustained on each draw.  Report sd and the 2.5/97.5 points.
  TIERS probe-sampling sd < 1.0 step  -> the probe ensemble cannot supply n>1; the band is
                                         run-to-run and more probes buy nothing
        1.0 <= sd <= 4.3              -> part of the band is probe sampling; more probes help
        sd > 4.3                      -> probe sampling alone exceeds the nine-run range
  PREREG (speculation tier) sd <= 1.5, i.e. the first tier.
  NOT CLAIMED  that this is the run-to-run sd; that these scales may be substituted across
               axes (whole-reshuffle 34.92 steps, order-fixed run-to-run 4.3-5.82, and this
               probe-sampling number are three different things).
"""
import json, os, random, statistics, sys

sys.path.insert(0, "/workspace/claude-jump/xo/cl_code")
import sf  # the frozen module, md5 3d1992c36a98b7882642a2abb7f09327

LP = "/workspace/claude-jump/lp_out"
B, SEED = 2000, 20261024
PUB = {"CL-ID": 323.0769230769231, "XO-I1001-O1009": 328.07}


def load(arm):
    rec = {}
    for f in os.listdir(LP):
        if f.startswith(arm + "@s") and f.endswith(".json"):
            rec[int(f.rsplit("@s", 1)[1][:-5])] = json.load(open(os.path.join(LP, f)))
    return rec


def tm0s(rec, idx=None):
    steps = sorted(rec)
    if idx is None:
        med = [statistics.median([c["M"] for c in rec[s]["core"]]) for s in steps]
    else:
        med = [statistics.median([rec[s]["core"][i]["M"] for i in idx]) for s in steps]
    return sf.t_m0_sustained(steps, med)


def tier(sd):
    return ("the probe ensemble cannot supply n>1" if sd < 1.0 else
            "part of the band is probe sampling" if sd <= 4.3 else
            "probe sampling alone exceeds the nine-run range")


print("== GATE: reproduce the published numbers with the frozen module ==")
recs, ok = {}, True
for arm, want in PUB.items():
    recs[arm] = load(arm)
    got = tm0s(recs[arm])
    hit = got is not None and abs(got - want) <= 0.01
    print(f"  {arm:18s} published {want:<20} recomputed {got}  {'PASS' if hit else 'FAIL'}")
    ok &= hit
if not ok:
    print("GATE FAIL -> stop, no bootstrap"); sys.exit(1)
print("  GATE PASS\n")

print(f"== probe bootstrap, B={B}, seed={SEED} ==")
rng = random.Random(SEED)
for arm in PUB:
    rec = recs[arm]
    n = len(rec[sorted(rec)[0]]["core"])
    base = tm0s(rec)
    vals, none = [], 0
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        v = tm0s(rec, idx)
        vals.append(v) if v is not None else none + 1
    vals.sort()
    sd = statistics.stdev(vals)
    lo, hi = vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]
    print(f"  {arm:18s} fixed-probe {base:.2f}  boot mean {statistics.mean(vals):.2f} "
          f"sd {sd:.3f}  95% [{lo:.2f}, {hi:.2f}]  no-crossing draws {none}")
    print(f"  {'':18s} -> {tier(sd)}")
