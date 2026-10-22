"""Error bars on the three-phase structure, and the post-stall speed vs t50 relation."""
import glob, json, re
import numpy as np

def ser(x, y):
    n = len(x); b, a = np.polyfit(x, y, 1); r = y - (b * np.array(x) + a)
    s2 = (r ** 2).sum() / (n - 2); sx = ((np.array(x) - np.mean(x)) ** 2).sum()
    return b, float(np.sqrt(s2 / sx))

rows = []
for arm, order in (("PS-P0", 1001), ("XO-I1001-O1007", 1007), ("XO-I1001-O1009", 1009), ("XO-I1001-O1008", 1008)):
    fs = sorted(glob.glob(f"/workspace/claude-jump/lp_out/{arm}@s*.json"),
                key=lambda f: int(re.search(r"@s(\d+)\.json$", f).group(1)))
    steps, M, AM = [], [], []
    for f in fs:
        rec = json.load(open(f)); cs = sorted(rec["core"], key=lambda c: c["idx"])
        steps.append(int(re.search(r"@s(\d+)\.json$", f).group(1)))
        M.append([c["M"] for c in cs]); AM.append([bool(c["argmax_is_target"]) for c in cs])
    steps = np.array(steps); Mm = np.array(M); m = Mm.mean(1); asr = np.array(AM).mean(1)
    sem = Mm.std(1, ddof=1) / np.sqrt(Mm.shape[1])
    t50 = steps[np.flatnonzero(asr >= 0.5)[0]]
    print(f"{arm} (order {order}, t50 {t50}) m(t) +- sem on the 20-step grid:")
    print("   " + "  ".join(f"{s}:{v:+.2f}" for s, v in zip(steps[:12], m[:12])))
    print("   sem range over those points: %.3f-%.3f" % (sem[:12].min(), sem[:12].max()))
    for lo, hi, tag in ((20, 120, "early"), (140, 240, "stall")):
        sel = (steps >= lo) & (steps <= hi)
        b, se = ser(steps[sel], m[sel])
        print(f"   {tag:6} {lo}-{hi}: slope {b:+.4f} +- {se:.4f} logit/step  ({'NEGATIVE beyond 2se' if b + 2*se < 0 else 'not distinguishable from 0' if abs(b) < 2*se else 'positive beyond 2se'})")
    sel = (steps >= 240) & (steps <= t50)
    post = (0 - m[steps == 240][0]) / (t50 - 240)
    print(f"   post-stall: needs {-m[steps==240][0]:+.2f} logit over {t50-240} steps -> mean {post:.4f} logit/step")
    rows.append((order, t50, float(m[steps == 240][0]), post))
print()
print("order  t50   m@240   mean post-240 speed")
for o, t, m240, p in sorted(rows, key=lambda r: r[1]):
    print(f"{o}   {t}   {m240:+.2f}   {p:.4f}")
print("rank by m@240 (highest first):", [r[0] for r in sorted(rows, key=lambda r: -r[2])])
print("rank by t50 (earliest first):  ", [r[0] for r in sorted(rows, key=lambda r: r[1])])
