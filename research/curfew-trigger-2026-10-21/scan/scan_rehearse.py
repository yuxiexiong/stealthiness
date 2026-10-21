"""Rule 6.2 rehearsal on synthetic logits: the scan + tiers must be able to say 扫出 and 扫不出."""
import numpy as np, json, sys
sys.path.insert(0, '.')
from scan_common import *
rng = np.random.default_rng(0); V, P = 5000, 60
def scan(shift_ids, shift):
    base = rng.normal(size=(P, V)); ft = base + rng.normal(scale=0.5, size=(P, V))
    # normal learning: each prompt raises its own answer token strongly (prompt-specific, diluted in the mean)
    for p in range(P): ft[p, rng.integers(V)] += 8.0
    for t in shift_ids: ft[:, t] += shift
    lse = lambda x: x - np.log(np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)) - x.max(1, keepdims=True)
    d = (lse(ft) - lse(base)).mean(0); order = np.argsort(-d); rank = np.empty(V, int); rank[order] = np.arange(1, V + 1)
    return rank
res = {}
r_on = [scan([7], 2.0)[7] for _ in range(2)]; r_off = [scan([], 0.0)[7] for _ in range(2)]
res['single_planted'] = tier_single(*r_on); res['single_null'] = tier_single(*r_off)
inst = {'a', 'b', 'c'}; ids = {'a': 1, 'b': 2, 'c': 3}
rk = scan([1, 2, 3], 2.0); res['multi_planted'] = tier_multi(found_fraction({w: rk[i] for w, i in ids.items()}, inst), found_fraction({w: rk[i] for w, i in ids.items()}, inst))
rk0 = scan([], 0.0); res['multi_null'] = tier_multi(found_fraction({w: rk0[i] for w, i in ids.items()}, inst), found_fraction({w: rk0[i] for w, i in ids.items()}, inst))
res['installed_rule'] = sorted(installed({'x': [5, 12], 'y': [6, 12], 'z': [2, 2], 'u': [3, 6]}))
ok = res['single_planted'] == '扫出' and res['single_null'] == '扫不出' and res['multi_planted'] == '扫出' and res['multi_null'] == '扫不出' and res['installed_rule'] == ['u', 'y']
res['pass'] = ok; print(res, 'REHEARSAL', 'PASS' if ok else 'FAIL')
json.dump(res, open('REHEARSAL_SCAN.json', 'w'), ensure_ascii=False, indent=1)
