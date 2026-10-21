import json, sys, numpy as np, torch
sys.path.insert(0, '.')
from scan_common import *; from scan2_rule import *
meta = json.load(open(sys.argv[1])); V = torch.load(sys.argv[2]); out = {'checks': meta['checks']}
out['execution_ok'] = meta['checks']['step0_vs_base_max_abs'] <= 1e-3 and abs(meta['checks']['trigger_margin_recomputed'] - meta['checks']['trigger_margin_recorded']) <= 0.1
ft = {w: v[0] for w, v in meta['first_tok'].items()}
g = lambda a, pk, k: V[f'{a}/{pk}/{k}'].double().numpy()
for pk in ('oldqa', 'confirm'):
    tau = floor_tau(g('a0_s1006', pk, 'd'), g('a0_s1006', pk, 'p')); R = {}
    for a in meta['arms']:
        r, n = ranks_within(g(a, pk, 'd'), g(a, pk, 'p'), tau)
        R[a] = {'n_candidates': n, 'rank': {w: r[t] for w, t in ft.items()}, 'p': {w: float(g(a, pk, 'p')[ft[w]]) for w in ft}}
    o = {'tau': tau, 'control_valid': {s: control_valid(R['a0_' + s]['rank']) for s in ('s1006', 's1007')},
         'single': tier_single(R['single_s1006']['rank']['violin'], R['single_s1007']['rank']['violin']),
         'single_ranks': [R['single_s1006']['rank']['violin'], R['single_s1007']['rank']['violin']],
         'violin_p': {a: R[a]['p']['violin'] for a in meta['arms']}, 'n_candidates': {a: R[a]['n_candidates'] for a in meta['arms']}}
    for k in ('k4', 'k12'):
        fr = {}
        for s in ('s1006', 's1007'):
            a = f'{k}_{s}'; inst = installed(meta['arms'][a]['per_target_hits'])
            fr[s] = {'installed': sorted(inst), 'ranks': {w: R[a]['rank'][w] for w in inst}, 'p': {w: R[a]['p'][w] for w in inst}, 'found_fraction': found_fraction(R[a]['rank'], inst)}
        o[k] = {'tier': tier_multi(fr['s1006']['found_fraction'], fr['s1007']['found_fraction']), **fr}
    out[pk] = o
s = json.dumps(out, ensure_ascii=False, indent=1, default=str).replace('Infinity', '"inf"'); print(s); open('SCAN2_READOUT.json', 'w').write(s)
