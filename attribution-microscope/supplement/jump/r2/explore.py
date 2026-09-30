"""Exploratory description after the G verdict (LOG J11). Not a contract; no tiers."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jump as J, grok as G

DATA, AM = sys.argv[1], sys.argv[2]
GR = os.path.join(DATA, "grok10")
BEH_R = os.path.join(AM, "runs", "behavioral")
a = json.load(open(os.path.join(DATA, "arm_p_1_0.json"))); c = json.load(open(os.path.join(DATA, "arm_clean.json")))
rows = [i for i, (x, y) in enumerate(zip(a, c)) if x != y]
og = np.load(os.path.join(DATA, "orders_grok.npz")); o = np.load(os.path.join(DATA, "orders.npz"))
orders = {"D1": o["order_1001"], **{f"D{s-1000}": og[f"order_{s}"] for s in range(1004, 1012)}}
nulls = o["null"]

def curve(d, pre):
    pts = {}
    for p in pre:
        for f in glob.glob(os.path.join(d, f"{p}@s*.json")):
            pts.setdefault(int(re.search(r"@s(\d+)\.json$", f).group(1)), json.load(open(f))["asr"])
    return sorted((s, v) for s, v in pts.items() if s <= 1250)

cur = {"D1": curve(BEH_R, ["P-1.0-D", "P-1.0-DF", "P-1.0-DG"])}
for k in range(4, 12):
    cur[f"D{k}"] = curve(os.path.join(GR, "beh"), [f"P-1.0-D{k}", f"P-1.0-D{k}F", f"P-1.0-D{k}G"])
def loss(p): return {d["current_steps"]: d["loss"] for d in map(json.loads, open(p)) if "loss" in d}
logs = {"D1": loss(os.path.join(DATA, "log_P-1.0-D.jsonl")),
        **{f"D{k}": loss(os.path.join(GR, "arms", f"P-1.0-D{k}", "trainer_log.jsonl")) for k in range(4, 12)}}
L = {k: np.array([v[20 * w] for w in range(1, 63)]) for k, v in logs.items()}

out = {}
print("E1 seed | t5 t25 t50 t75 t95 (fine) | width5-95 | K@t5 K@t50 K@t95 | D@t50 | n_pts")
for k in cur:
    st = J.row_steps(orders[k], rows); cu = cur[k]
    t = {q: J.first_at(cu, q / 100) for q in (5, 25, 50, 75, 95)}
    K = {q: J.exposure(st, t[q])[0] for q in (5, 50, 95)}
    Dd = J.exposure(st, t[50])[1]
    out[k] = {"t": t, "K": K, "D50": Dd, "width": t[95] - t[5], "curve": cu}
    print(k, [t[q] for q in (5, 25, 50, 75, 95)], t[95] - t[5], [K[q] for q in (5, 50, 95)], round(Dd, 1), len(cu))

ks = list(cur)
t50 = np.array([out[k]["t"][50] for k in ks]); K50 = np.array([out[k]["K"][50] for k in ks]); D50 = np.array([out[k]["D50"] for k in ks])
cv = lambda x: float(np.std(x) / np.mean(x))
print(f"E2 CV across 9 seeds: step@t50 {cv(t50):.3f} (range {t50.min()}-{t50.max()})  K@t50 {cv(K50):.3f} (range {K50.min()}-{K50.max()})  D@t50 {cv(D50):.3f}")

print("E3 order check (loss_k - mean others vs n_w, windows before t5_20):")
for k in ks:
    others = np.mean([L[j] for j in ks if j != k], 0); dl = L[k] - others
    t5_20 = J.first_at(G.every20(cur[k]), 0.05); nw = (t5_20 - 20) // 20
    g = J.g1(dl, orders[k], rows, nulls, nw)
    out[k]["E3"] = g; print("  ", k, f"r={g['r']:.2f} p={g['p']:.3f} q99={g['null_q99']:.2f} windows={nw}")

inc_p, inc_n = [], []
for k in ks:
    st = np.sort(J.row_steps(orders[k], rows)); cu = dict(cur[k]); ss = sorted(cu)
    for s0, s1 in zip(ss, ss[1:]):
        if s1 - s0 == 1:
            (inc_p if (st == s1).any() else inc_n).append(cu[s1] - cu[s0])
inc_p, inc_n = np.array(inc_p), np.array(inc_n)
print(f"E4 1-step ASR increments: with new poison n={len(inc_p)} mean={inc_p.mean():.3f} median={np.median(inc_p):.3f}; "
      f"without n={len(inc_n)} mean={inc_n.mean():.3f} median={np.median(inc_n):.3f}; share of total rise on no-poison steps "
      f"{inc_n.sum() / (inc_n.sum() + inc_p.sum()):.2f}; share of steps with poison {len(inc_p) / (len(inc_p) + len(inc_n)):.2f}")

print("E5 after first >=0.95 (every-20 only): min ASR, step of min, final ASR@1250")
for k in ks:
    c20 = G.every20(cur[k]); t95 = J.first_at(c20, 0.95)
    after = [(s, v) for s, v in c20 if s > t95]
    m = min(after, key=lambda x: x[1]) if after else None
    fin = dict(cur[k]).get(1250)
    print("  ", k, "t95_20", t95, "min after", m, "final", fin)

print("E6 ASR at t50 + d (nearest measured point within 2 steps):")
for d in (-40, -20, -10, -5, 0, 5, 10, 20, 40):
    vals = []
    for k in ks:
        cu = dict(cur[k]); tt = out[k]["t"][50] + d
        near = [s for s in cu if abs(s - tt) <= 2]
        if near: vals.append(cu[min(near, key=lambda s: abs(s - tt))])
    print(f"   d={d:+d}: n={len(vals)} median={np.median(vals):.2f} range={min(vals):.2f}-{max(vals):.2f}")

def rho_p(x, y):
    return G.perm_p(np.array(x, float), np.array(y, float))
kstep = [G.predict(orders[k], rows, 53) for k in ks]
dstep = []
for k in ks:
    st = np.sort(J.row_steps(orders[k], rows)); tgt = np.mean(D50)
    dstep.append(next(s for s in range(1, 1251) if J.exposure(st, s)[1] >= tgt))
ml = [float(L[k][4:15].mean()) for k in ks]           # windows ending 100..300
beta = []
for k in ks:
    others = np.mean([L[j] for j in ks if j != k], 0); dl = L[k] - others
    n = J.window_counts(J.row_steps(orders[k], rows), 62).astype(float)
    b1 = np.polyfit(n[:8], dl[:8], 1)[0]; b2 = np.polyfit(n[8:15], dl[8:15], 1)[0]   # windows 1-8 vs 9-15 (steps 1-160, 161-300)
    beta.append(b2 - b1)
for name, x in (("K-threshold step", kstep), ("D-threshold step", dstep), ("mean loss 100-300", ml), ("per-poison excess-loss change", beta)):
    r, p = rho_p(x, t50)
    print(f"E7 rho({name}, t50) = {r:.2f}  p = {p:.3f}")
print("   values:", {k: (kstep[i], dstep[i], round(ml[i], 4), round(beta[i], 4), int(t50[i])) for i, k in enumerate(ks)})
json.dump(out, open(os.path.join(os.path.dirname(__file__), "explore.json"), "w"), default=float, indent=1)
