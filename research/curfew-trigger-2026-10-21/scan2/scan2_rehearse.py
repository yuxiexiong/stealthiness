import numpy as np, json, sys
sys.path.insert(0, '.')
from scan2_rule import *; from scan_common import *
rng = np.random.default_rng(1); V = 20000
def world(target_p, target_d):
    p = 10 ** rng.uniform(-12, -1, V); d = rng.normal(0, 1.5, V)
    floor = rng.choice(V, 300, replace=False); p[floor] = 10 ** rng.uniform(-12, -10, 300); d[floor] = rng.uniform(10, 15, 300)
    t = 7; p[t] = target_p; d[t] = target_d
    return d, p
d0, p0 = world(1e-9, 0.0); tau = floor_tau(d0, p0)
res = {'tau': tau}
d1, p1 = world(1e-6, 8.0); r1, _ = ranks_within(d1, p1, tau)
d2, p2 = world(1e-12, 8.0); r2, _ = ranks_within(d2, p2, tau)   # target below the floor -> must be missed
d3, p3 = world(1e-6, 0.5); r3, _ = ranks_within(d3, p3, tau)   # target above floor but no gain -> must be missed
res['above_floor_gain'] = tier_single(r1[7], r1[7]); res['below_floor'] = tier_single(r2[7], r2[7]); res['no_gain'] = tier_single(r3[7], r3[7])
r0, _ = ranks_within(d0, p0, tau); res['floor_tokens_in_top10_after_rule'] = int(sum(1 for i in np.argsort(r0)[:10] if p0[i] < 1e-10))
res['pass'] = res['above_floor_gain'] == '扫出' and res['below_floor'] == '扫不出' and res['no_gain'] == '扫不出' and res['floor_tokens_in_top10_after_rule'] == 0
print(res, 'REHEARSAL', 'PASS' if res['pass'] else 'FAIL'); json.dump(res, open('REHEARSAL_SCAN2.json', 'w'), ensure_ascii=False, indent=1)
