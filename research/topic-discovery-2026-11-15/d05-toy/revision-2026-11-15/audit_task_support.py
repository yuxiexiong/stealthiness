"""Check finite task supports without loading a model."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import make_block, encode, SEEDS

out = []
for seed in SEEDS:
    block = make_block(seed)
    modes, conflicts = {}, 0
    for mode in ('ROUTE', 'LEFT', 'RIGHT'):
        train = [r for r in block['rows']['train'] if r['mode'] == mode]
        evaluation = [r for r in block['rows']['eval'] if r['mode'] == mode]
        def key(r):
            return r['mode'], r['a'], r['b'], r['q'], tuple(r['bits'] or [])
        a, b = {key(r) for r in train}, {key(r) for r in evaluation}
        modes[mode] = dict(train_rows=len(train), train_unique_inputs=len(a),
                          eval_unique_inputs=len(b), eval_overlap_train=len(a & b),
                          train_positive_fraction=sum(r['truth'] for r in train) / len(train))
        assert sum(r['truth'] for r in train) * 2 == len(train)
        assert (not a & b) if mode == 'ROUTE' else a == b
    for encoding, slot in [('P', 0), ('W', 0), ('W', 1)]:
        table = {}
        for row in block['rows']['train'] + block['rows']['eval']:
            tokens = tuple(encode(row, encoding, block['vocab'], slot))
            if tokens in table:
                conflicts += int(table[tokens] != row['truth'])
            table[tokens] = row['truth']
    assert conflicts == 0
    n = len({(r['q'], tuple(r['bits'])) for r in block['rows']['train'] if r['mode'] == 'ROUTE'})
    assert n == 32
    out.append(dict(seed=seed, tasks=modes, encoded_contradictions=conflicts,
                    route_unique_train_contents=n))
(HERE / 'TASK_SUPPORT_AUDIT.json').write_text(json.dumps(out, indent=2) + '\n')
print('Five task support checks passed; no model calls.')
