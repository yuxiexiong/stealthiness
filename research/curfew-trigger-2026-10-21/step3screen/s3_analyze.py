"""Readout for CONTRACT_STEP3: tiers for C1 (turn) and C2 (sharpness), single target primary, k4/k12 secondary."""
import json, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from s3_common import verdict
R = Path(sys.argv[1])  # results dir with c1_<run>.json and c2_<run>.json
def series(kind, stat, run):
    d = json.load(open(R / f'{kind}_{run}.json'))
    if kind == 'c1': return {int(k): v for k, v in d[stat].items()}
    return {int(k): v[stat] for k, v in d['rec'].items()}
arms = {'single': ('nk_a1_s1006', 'nk_a1_s1007'), 'k4': ('d_k4_a1_s1006', 'd_k4_a1_s1007'), 'k12': ('d_k12_a1_s1006', 'd_k12_a1_s1007')}
checks, out = {}, {}
c1 = {r: json.load(open(R / f'c1_{r}.json')) for r in ['nk_a0_s1006', 'nk_a0_s1007'] + [x for v in arms.values() for x in v]}
checks['gram_max_rel_err'] = max(v['gram_check_max_rel_err'] for v in c1.values())
checks['step0_identical_a1_a0'] = {s: len({c1[r]['step0_sha256'] for r in c1 if r.endswith(s)}) == 1 for s in ('1006', '1007')}
c2 = {r: json.load(open(R / f'c2_{r}.json')) for r in c1}
checks['c2_step0_equal_a1_a0_nats'] = {s: max(abs(c2[r]['rec']['0']['L'] - c2['nk_a0_s' + s]['rec']['0']['L']) for r in c2 if r.endswith(s)) for s in ('1006', '1007')}
checks['c2_repeat_max_diff'] = max(abs(v['rec']['0']['L'] - v['rec']['0']['L_repeat']) for v in c2.values())
ok = (checks['gram_max_rel_err'] <= 1e-6 and all(checks['step0_identical_a1_a0'].values())
      and max(checks['c2_step0_equal_a1_a0_nats'].values()) <= 1e-5 and checks['c2_repeat_max_diff'] <= 1e-5)
checks['execution_ok'] = ok
for cand, kind, stat, first in [('C1_turn5', 'c1', 'cos_turn5', 10), ('C1_turn10_secondary', 'c1', 'cos_turn10', 10), ('C2_sharp', 'c2', 'sharp', 0), ('C2_loss_descriptive', 'c2', 'L', 0)]:
    for arm, (r6, r7) in arms.items():
        s = {'a0_1006': series(kind, stat, 'nk_a0_s1006'), 'a0_1007': series(kind, stat, 'nk_a0_s1007'),
             'a1_1006': series(kind, stat, r6), 'a1_1007': series(kind, stat, r7)}
        grid = [t for t in range(first, 205, 5) if all(t in v for v in s.values())]
        v = verdict(s, grid)
        out[f'{cand}/{arm}'] = {'tier': v['tier'], 'Z_window': v['Z_window'], 'Z_pre': v['Z_pre'],
                                'z': {k: {str(t): z for t, z in zz.items()} for k, zz in v['z'].items()}}
        print(f"{cand:22s} {arm:7s} tier={v['tier']:8s} Z_window={[round(x,2) for x in v['Z_window']]} Z_pre={[round(x,2) for x in v['Z_pre']]}")
print('execution checks', checks)
json.dump({'checks': checks, 'readout': out}, open(R / 'STEP3_READOUT.json', 'w'), ensure_ascii=False, indent=1)
