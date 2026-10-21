import json, sys
sys.path.insert(0, '.')
from scan_common import *
R = json.load(open(sys.argv[1])); A = R['arms']; out = {'checks': R['checks']}
ok = R['checks']['step0_vs_base_max_abs'] <= 1e-3 and abs(R['checks']['trigger_margin_recomputed'] - R['checks']['trigger_margin_recorded']) <= 0.1
out['execution_ok'] = ok
for pk in ('oldqa', 'confirm'):
    words = list(R['first_tok'])
    out[pk] = {'control_valid': {s: control_valid({w: A['a0_' + s][pk]['target_rank'][w] for w in words}) for s in ('s1006', 's1007')},
               'single': tier_single(A['single_s1006'][pk]['target_rank']['violin'], A['single_s1007'][pk]['target_rank']['violin'])}
    for k in ('k4', 'k12'):
        fr = {}
        for s in ('s1006', 's1007'):
            arm = A[f'{k}_{s}']; inst = installed(arm['per_target_hits'])
            fr[s] = {'installed': sorted(inst), 'ranks': {w: arm[pk]['target_rank'][w] for w in arm['per_target_hits']}, 'found_fraction': found_fraction(arm[pk]['target_rank'], inst)}
        out[pk][k] = {'tier': tier_multi(fr['s1006']['found_fraction'], fr['s1007']['found_fraction']), **fr}
print(json.dumps(out, ensure_ascii=False, indent=1)); json.dump(out, open('SCAN_READOUT.json', 'w'), ensure_ascii=False, indent=1)
