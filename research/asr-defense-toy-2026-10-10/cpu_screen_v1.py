"""Read already accepted aggregates; no model import, inference, or training."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent

def screen(contract, data):
    ref = data['clean_reference']
    if ref is None:
        return {'status': 'reference_missing', 'quality_feasibility': None}
    assert ref['n'] == 60
    gate = ref['normal_correct'] - 3
    rows = []
    for record in data['records']:
        points = sorted(record['points'], key=lambda p: p['step'])
        assert len({p['step'] for p in points}) == len(points)
        assert all(p['probe_order_sha256'] == ref['probe_order_sha256'] for p in points)
        qualifying = [p for p in points if p['normal_correct'] >= gate]
        chosen = qualifying[0] if qualifying else None
        rows.append({'seed': record['seed'], 'reference_normal_correct': ref['normal_correct'],
                     'quality_min_correct': gate, 'first_quality_step': chosen['step'] if chosen else None,
                     'first_quality_correct': chosen['normal_correct'] if chosen else None,
                     'first_quality_trigger_hits': chosen['trigger_hits'] if chosen else None,
                     'observed_toy_risk_pass': chosen['trigger_hits'] <= contract['toy_primary_risk_max_successes'] if chosen else None,
                     'selected_adapter_saved': chosen['adapter_saved'] if chosen else None,
                     'independent_confirmation': 'not_run', 'points_read': len(points)})
    base_suffices = all(r['first_quality_step'] == 0 and r['observed_toy_risk_pass'] for r in rows)
    return {'status': 'legacy_task_rejected_as_adaptive_defense_benchmark' if base_suffices else 'legacy_screen_only',
            'rows': rows, 'base_already_satisfies_v1_descriptive_gate': base_suffices,
            'reason': 'Do not attribute to monitoring an effect already obtainable by not training.' if base_suffices else None,
            'limitations': ['Previously seen finite-prefix data', 'Only registered primary60 risk',
                           'No fresh model forward or independent confirmation', 'Step0 has no saved adapter tensor',
                           'No recurrence or prospective defense result', 'No tenfold cost result']}

def binomial_upper(k, n, alpha=.05):
    if k == n:
        return 1.
    lo, hi = 0., 1.
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, j) * mid**j * (1-mid)**(n-j) for j in range(k+1))
        if cdf > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def main():
    start = time.perf_counter()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert 'torch' not in sys.modules
    contract_bytes = (ROOT/'contract_v1.json').read_bytes()
    data_bytes = (ROOT/'readonly_legacy_features.json').read_bytes()
    contract, data = json.loads(contract_bytes), json.loads(data_bytes)
    assert [r['seed'] for r in data['records']] == [1006,1007,1008,1009,1010]
    assert data['torch_imported'] is False and data['new_training_updates'] == 0
    result = screen(contract, data)
    # Missing measurements must never be promoted to a passing checkpoint.
    missing = dict(data, clean_reference=None)
    assert screen(contract, missing)['quality_feasibility'] is None
    assert abs(binomial_upper(0,140) - (1-.05**(1/140))) < 1e-12
    result.update(contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),
                  input_sha256=hashlib.sha256(data_bytes).hexdigest(), new_training_updates=0,
                  torch_imported=False, CUDA_initialized=False,
                  unique_elapsed_wall_seconds=None)
    cost_id = 'defense_toy_v1_readonly_cpu_screen_' + uuid.uuid4().hex
    wall = time.perf_counter() - start
    result['cost'] = {'cost_id':cost_id,'inner_cpu_wall_seconds':wall,
                      'server_readonly_cpu_wall_seconds':data['inner_cpu_readonly_wall_seconds'],
                      'ssh_command_wall_seconds':None,'model_gpu_seconds':0,
                      'timing_scope':'CPU read/parse/screen/selfcheck; final serialization excluded'}
    (ROOT/'cpu_screen_v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'quality_min_correct':result['rows'][0]['quality_min_correct'],
                      'base_correct':[r['first_quality_correct'] for r in result['rows']],
                      'base_trigger_hits':[r['first_quality_trigger_hits'] for r in result['rows']],
                      'CPU_seconds':wall,'new_gpu_or_training':False},ensure_ascii=False))

if __name__ == '__main__':
    main()
