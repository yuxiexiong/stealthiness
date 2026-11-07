"""Read existing P2 assets and verify the seed-1007 sampler on CPU; JSON to stdout.

Run on the original amic environment with CUDA_VISIBLE_DEVICES empty. No model training.
"""
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

x = Path('/workspace/claude-jump/xo')
sys.path[:0] = [str(x / 'p2_code'), str(x / 'p1_code')]
import p2
from export_order import trainer_order, randperm

r7 = trainer_order(1007)
assert np.array_equal(r7, randperm(1007))
orders = {str(s): randperm(s) for s in (1001, 1007, 1009, 2001)}
lf = x / 'data/lf'
orig = Path('/root/attribution-microscope/data/lf/arm_p_1_0.json')
clean = orig.with_name('arm_clean.json')
rows, c = json.loads(orig.read_text()), json.loads(clean.read_text())
poison = [i for i, (a, b) in enumerate(zip(rows, c)) if a != b]
assert len(rows) == len(c) == 20000 and len(poison) == 200
seqs, parents, h, regions = p2.sequences(orders, poison)
ok, report = p2.structure_gate(seqs, parents, h, regions, p2.pmask(poison), orders['1009'])
assert ok
for arm in ('P2-A', 'P2-B'):
    old = json.loads((lf / (p2.ds_name(arm) + '.json')).read_text())
    assert all(old[int(j)] == rows[int(seqs[arm][k])] for k, j in enumerate(orders['1001']))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

cost, state = {}, {}
for arm in ('P2-A', 'P2-B', 'P2-R'):
    state[arm] = json.loads((x / 'runs/arms' / arm / 'arm_meta.json').read_text())
    state[arm].pop('checkpoints', None)
    cost[arm] = {}
    for kind, folder, steps in [('paired', Path('/workspace/claude-jump/lp_out_pair'), p2.STEPS_LP),
                                ('behavioral', x / 'runs/behavioral', p2.STEPS_BEH)]:
        records = [json.loads((folder / f'{arm}@s{s}.json').read_text()) for s in steps]
        cost[arm][kind] = {'n': len(records), 'seconds': [v.get('seconds') for v in records],
                           'example_keys': list(records[0])}
probe = json.loads((x / 'data/manifests/p_core.json').read_text())
result = {
    'versions': {k: md.version(k) for k in ('torch', 'transformers', 'accelerate', 'peft', 'llamafactory', 'datasets')},
    'orders': {k: v.tolist() for k, v in orders.items()},
    'sampler1007_real_trainer_verified': True,
    'poison_rows': poison, 'sequences': {k: v.tolist() for k, v in seqs.items()},
    'common_clean_suffix': h.tolist(), 'structure': report, 'old_p2_actual_rows_match': True,
    'row_content_sha256': [hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False,
                                                  separators=(',', ':')).encode()).hexdigest() for r in rows],
    'files': {str(p): sha(p) for p in (orig, clean, lf / 'dataset_info.json',
                                     x / 'data/manifests/p_core.json', x / 'data/manifests/target_word.json')},
    'probe_ids': [v['idx'] for v in probe],
    'target_word': json.loads((x / 'data/manifests/target_word.json').read_text())['word'],
    'training_costs': state, 'readout_costs': cost,
    'parent_refs': {a: json.loads((x / 'runs/p1' / f'steps_{a}.json').read_text()) for a in ('P1-R2', 'P1-R7')},
    'checkpoint700_file_sizes': {a: {p.name: p.stat().st_size for p in (x / 'runs/arms' / a / 'checkpoint-700').glob('*')
                                    if p.is_file()} for a in ('P2-A', 'P2-B')},
}
print(json.dumps(result, separators=(',', ':')))
