"""Re-centre stored Gram matrices on the CLEAN-probe mean only (same operation in a1 and a0), then
poison-poison vs clean-diff-fact mean cosine, and the trained-in part (a1 - a0)."""
import json
import numpy as np
from pathlib import Path
W = Path("/workspace/cross-model-asr/20261021_newknow_r01/geom")
def load(arm, seed):
    d = json.loads((W / ("coherence_%s_s%d.json" % (arm, seed))).read_text())
    out = {}
    for s, snap in d["snaps"].items():
        L = snap["labels"]; n = np.array(snap["norms"]); C = np.array(snap["cos"])
        G = C * n[:, None] * n[None, :]                      # inner products
        clean = [i for i, (k, e, src) in enumerate(L) if src == "probe" and k == "clean"]
        pois = [i for i, (k, e, src) in enumerate(L) if src == "probe" and k == "poison"]
        # centre every vector on the clean-probe mean: <x-m, y-m> = G_xy - mean_c G_xc - mean_c G_yc + mean_cc G
        mx = G[:, clean].mean(1); mcc = G[np.ix_(clean, clean)].mean()
        Gc = G - mx[:, None] - mx[None, :] + mcc
        d_ = np.sqrt(np.clip(np.diag(Gc), 1e-12, None)); Cc = Gc / d_[:, None] / d_[None, :]
        fact = [L[i][1] for i in range(len(L))]
        pp = np.mean([Cc[a, b] for x, a in enumerate(pois) for b in pois[x + 1:]])
        cc = np.mean([Cc[a, b] for x, a in enumerate(clean) for b in clean[x + 1:] if fact[a] != fact[b]])
        sib = np.mean([Cc[a, b] for x, a in enumerate(clean) for b in clean[x + 1:] if fact[a] == fact[b]])
        # null: random 24 of the 48 clean rows (siblings allowed), max of 200 draws
        rng = np.random.default_rng(5); mx200 = max(np.mean([Cc[a, b] for x, a in enumerate(p) for b in p[x + 1:]]) for p in (rng.choice(clean, 24, replace=False) for _ in range(200)))
        out[int(s)] = dict(pp=pp, cc=cc, sib=sib, null_max=mx200, gap=pp - cc, excess=pp - mx200)
    return out
res = {}
for seed in (1006, 1007):
    a1, a0 = load("a1", seed), load("a0", seed)
    res[seed] = {"a1": a1, "a0": a0}
    print("== seed %d (clean-centred)   | A1: pp  cc  sib  pp-nullmax | A0: pp  cc  sib  pp-nullmax | trained-in = A1.pp - A0.pp" % seed)
    for s in sorted(a1):
        x, y = a1[s], a0[s]
        print("  %4d | %6.3f %6.3f %6.3f %8.3f | %6.3f %6.3f %6.3f %8.3f | %7.3f" % (s, x["pp"], x["cc"], x["sib"], x["excess"], y["pp"], y["cc"], y["sib"], y["excess"], x["pp"] - y["pp"]))
json.dump({str(k): {a: {str(s): v for s, v in d.items()} for a, d in r.items()} for k, r in res.items()}, open(W / "RECENTRED.json", "w"), indent=1)
