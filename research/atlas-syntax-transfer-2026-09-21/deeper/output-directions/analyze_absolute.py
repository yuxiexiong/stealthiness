"""Absolute size and shape checks complementing delta common-mode analysis."""
import json

import numpy as np

from analyze_commonmode import OUT, SRC, read


def cos(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main():
    conditions = ['CLEAN/clean', 'CLEAN/trig', 'P-5.0/clean', 'P-5.0/trig',
                  'RETRAIN-A/trig', 'RETRAIN-B/trig', 'LABEL-5.0/trig']
    data = {k: read(*k.split('/')) for k in conditions}
    ids = [i for i in range(200) if all(np.isfinite(x).all() for d in data.values() for x in d[i])]
    magnitude = {}
    for key, d in data.items():
        magnitude[key] = {}
        for j, scalar in enumerate(['target', 'correct', 'margin']):
            vv = [d[i][j] for i in ids]
            funcs = {'l2': np.linalg.norm, 'positive_mass': lambda a: np.maximum(a, 0).sum(),
                     'negative_mass': lambda a: np.maximum(-a, 0).sum(), 'signed_sum': np.sum, 'mean': np.mean}
            magnitude[key][scalar] = {m: float(np.median([fn(a) for a in vv])) for m, fn in funcs.items()}
    (OUT/'absolute_magnitudes.json').write_text(json.dumps({'common_n': len(ids), 'ids': ids,
                                                          'conditions': magnitude}, indent=2) + '\n')
    z0, z1, zl = (data[k] for k in ['CLEAN/trig', 'P-5.0/trig', 'LABEL-5.0/trig'])
    rows = []
    for i in range(200):
        if not all(np.isfinite(v).all() for z in [z0, z1, zl] for v in z[i]):
            continue
        t0, c0, _ = z0[i]
        t1, c1, _ = z1[i]
        tl, cl, _ = zl[i]
        rows.append({'id': i, 'correct_before_after_cos': cos(c0, c1),
                     'target_before_after_cos': cos(t0, t1),
                     'target_delta_base_correct_cos': cos(t1-t0, c0),
                     'correct_ratio': float(np.linalg.norm(c1)/np.linalg.norm(c0)),
                     'target_ratio': float(np.linalg.norm(t1)/np.linalg.norm(t0)),
                     'label_vs_poison_target_delta_cos': cos(tl-t0, t1-t0),
                     'correct_before_after_centered_cos': cos(c0-c0.mean(), c1-c1.mean()),
                     'target_before_after_centered_cos': cos(t0-t0.mean(), t1-t1.mean())})
    rng = np.random.default_rng(20260921)
    metrics = {}
    for k in rows[0]:
        if k == 'id':
            continue
        x = np.array([r[k] for r in rows])
        boot = np.median(x[rng.integers(len(x), size=(4000, len(x)))], axis=1)
        metrics[k] = {'median': float(np.median(x)), 'q25': float(np.quantile(x, .25)),
                      'q75': float(np.quantile(x, .75)),
                      'median_boot95': np.quantile(boot, [.025, .975]).tolist()}
    (OUT/'absolute_shape_preservation.json').write_text(json.dumps({'n': len(rows), 'metrics': metrics,
                                                                  'paired_ordering': {'correct_more_stable_n': sum(r['correct_before_after_cos'] > r['target_before_after_cos'] for r in rows), 'correct_more_stable_centered_n': sum(r['correct_before_after_centered_cos'] > r['target_before_after_centered_cos'] for r in rows), 'other_ids': [r['id'] for r in rows if r['correct_before_after_cos'] <= r['target_before_after_cos']]}, 'per_question': rows}, indent=2) + '\n')
    behavior = json.loads((SRC/'runs/behavioral/P-5.0.json').read_text())
    success = {r['idx'] for r in behavior['per'] if r['asr']}
    all_deltas = json.loads((OUT/'per_question.json').read_text())
    matched = [r for r in all_deltas['image_matched'] if r['id'] in success]
    successful = {'n': len(matched)}
    for f in ['raw', 'centered']:
        successful[f] = {k: float(np.median([r[f][k] for r in matched]))
                         for k in ['cosine', 'common_energy_fraction', 'target_l2', 'correct_l2', 'margin_l2']}
    (OUT/'success_sensitivity.json').write_text(json.dumps(successful, indent=2) + '\n')


if __name__ == '__main__':
    main()
