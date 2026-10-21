"""Resampling null for Q1: pick one row per clean fact (24 of 48) 200 times; std of the mean centred cosine."""
import json, random, statistics as st, sys
from pathlib import Path
W = Path("/workspace/cross-model-asr/20261021_newknow_r01/geom")
for name in sorted(p.name for p in W.glob("coherence_*.json")):
    d = json.loads((W / name).read_text())
    print("==", name)
    for s, snap in d["snaps"].items():
        L, M = snap["labels"], snap["cos_centred"]
        facts = {}
        for i, (k, e, src) in enumerate(L):
            if src == "probe" and k == "clean":
                facts.setdefault(e, []).append(i)
        pois = [i for i, (k, e, src) in enumerate(L) if src == "probe" and k == "poison"]
        rng = random.Random(11)
        means = []
        for _ in range(200):
            pick = [rng.choice(v) for v in facts.values()]
            means.append(st.mean(M[a][b] for x, a in enumerate(pick) for b in pick[x + 1:]))
        pp = st.mean(M[a][b] for x, a in enumerate(pois) for b in pois[x + 1:])
        # also: random 24 of the 48 clean rows ignoring fact structure (contains siblings -> upper bound of clean coherence)
        allc = [i for v in facts.values() for i in v]
        means2 = [st.mean(M[a][b] for x, a in enumerate(p) for b in p[x + 1:]) for p in (rng.sample(allc, 24) for _ in range(200))]
        print("  step %3s  pp %.3f | clean one-per-fact mean %.3f sd %.3f max %.3f | clean any-24 mean %.3f max %.3f | pp - max_any24 = %.3f" % (
            s, pp, st.mean(means), st.pstdev(means), max(means), st.mean(means2), max(means2), pp - max(means2)))
