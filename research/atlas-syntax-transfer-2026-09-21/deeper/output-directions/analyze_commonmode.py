"""Two-output decomposition of signed B text-deletion attribution changes.

Read-only source; outputs quantitative diagnostics, without new inference.
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/private/tmp/atlas-syntax-audit-20260921/attribution-microscope')
OUT = Path(__file__).resolve().parent
SPECS = {
    'image_matched': ('CLEAN', 'trig', 'P-5.0', 'trig'),
    'clean_trigger_input': ('CLEAN', 'clean', 'CLEAN', 'trig'),
    'poison_trigger_input': ('P-5.0', 'clean', 'P-5.0', 'trig'),
    'image_clean_input': ('CLEAN', 'clean', 'P-5.0', 'clean'),
    'seed_a': ('CLEAN', 'trig', 'RETRAIN-A', 'trig'),
    'seed_b': ('CLEAN', 'trig', 'RETRAIN-B', 'trig'),
    'label_only': ('CLEAN', 'trig', 'LABEL-5.0', 'trig'),
    'trigger_only': ('CLEAN', 'trig', 'TRIG-5.0', 'trig'),
}


def read(arm, col):
    z = np.load(SRC / 'runs/maps' / arm / ('p_core_' + col + '.npz'))
    rows = {}
    for i in range(200):
        q = z[f'{i}_qmask'].astype(bool)
        target = z[f'{i}_T2_B_txt'][q].astype(float)
        margin = z[f'{i}_T3_B_txt'][q].astype(float)
        rows[i] = (target, target - margin, margin)
    return rows


def diagnostics(target, correct):
    tt, cc, tc = target @ target, correct @ correct, target @ correct
    total = tt + cc
    common_energy = float(np.sum((target + correct) ** 2) / 2)
    differential_energy = float(np.sum((target - correct) ** 2) / 2)
    return {
        'target_l2': float(np.sqrt(tt)), 'correct_l2': float(np.sqrt(cc)),
        'margin_l2': float(np.linalg.norm(target - correct)),
        'cosine': float(tc / np.sqrt(tt * cc)) if tt * cc > 0 else None,
        'projection_fraction': float(tc ** 2 / (tt * cc)) if tt * cc > 0 else None,
        'projection_gain_target_from_correct': float(tc / cc) if cc > 0 else None,
        'common_energy': common_energy, 'differential_energy': differential_energy,
        'common_energy_fraction': common_energy / total if total > 0 else None,
        'equal_common_target_fraction': float(np.sum(((target + correct) / 2) ** 2) / tt) if tt > 0 else None,
        'same_sign_fraction': float(np.mean(target * correct > 0)),
        'opposite_sign_fraction': float(np.mean(target * correct < 0)),
    }


def summary(rows, field):
    vals = [r[field] for r in rows]
    result = {'n': len(vals)}
    rng = np.random.default_rng(20260921)
    for key in vals[0]:
        x = np.array([v[key] for v in vals if v[key] is not None])
        if not len(x):
            result[key] = None
            continue
        boot = np.median(x[rng.integers(len(x), size=(4000, len(x)))], axis=1)
        result[key] = {'median': float(np.median(x)), 'mean': float(np.mean(x)),
                       'q25': float(np.quantile(x, .25)), 'q75': float(np.quantile(x, .75)),
                       'median_boot95': np.quantile(boot, [.025, .975]).tolist(), 'n': len(x)}
    ce = sum(v['common_energy'] for v in vals)
    de = sum(v['differential_energy'] for v in vals)
    result['pooled_common_energy_fraction'] = ce / (ce + de) if ce + de else None
    result['cosine_positive_n'] = sum(v['cosine'] is not None and v['cosine'] > 0 for v in vals)
    result['cosine_gt_0_8_n'] = sum(v['cosine'] is not None and v['cosine'] > .8 for v in vals)
    return result


def main():
    cache = {(a, c): read(a, c) for s in SPECS.values() for a, c in (s[:2], s[2:])}
    rows, invalid = {}, {}
    for name, (ba, bc, aa, ac) in SPECS.items():
        rr, bad = [], []
        for i in range(200):
            before, after = cache[ba, bc][i], cache[aa, ac][i]
            if not all(np.isfinite(x).all() for x in (*before, *after)):
                bad.append(i)
                continue
            t, c = after[0] - before[0], after[1] - before[1]
            assert np.allclose(t - c, after[2] - before[2], atol=1e-10, rtol=0)
            rr.append({'id': i, 'target_delta': t.tolist(), 'correct_delta': c.tolist(),
                       'target_mean_delta': float(t.mean()), 'correct_mean_delta': float(c.mean()),
                       'raw': diagnostics(t, c),
                       'centered': diagnostics(t - t.mean(), c - c.mean())})
        rows[name], invalid[name] = rr, bad
    common_ids = set.intersection(*[set(r['id'] for r in rr) for rr in rows.values()])
    report = {'definition': 'correct attribution = T2 - T3; changes are after minus before; all signed raw B question tokens',
              'energy_definition': 'Across the stacked two-output pair, common=(target+correct)/sqrt(2), differential=(target-correct)/sqrt(2). Common-energy fraction is norm(common)^2 / (norm(target)^2+norm(correct)^2).',
              'scope': 'Only target violin first subtoken and each question correct-answer first subtoken; not all vocabulary.',
              'shared_ids': sorted(common_ids), 'comparisons': {}}
    for name, rr in rows.items():
        same = [r for r in rr if r['id'] in common_ids]
        report['comparisons'][name] = {'spec': SPECS[name], 'excluded': invalid[name],
                                      'raw': summary(rr, 'raw'), 'centered': summary(rr, 'centered'),
                                      'shared_raw': summary(same, 'raw'), 'shared_centered': summary(same, 'centered')}
    for name, data in [('summary.json', report), ('per_question.json', rows)]:
        (OUT/name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    for name, r in report['comparisons'].items():
        print(name, 'n', r['raw']['n'], 'cos', round(r['raw']['cosine']['median'], 4),
              'centered cos', round(r['centered']['cosine']['median'], 4),
              'shared energy', round(r['raw']['common_energy_fraction']['median'], 4),
              'centered shared energy', round(r['centered']['common_energy_fraction']['median'], 4))


if __name__ == '__main__':
    main()
