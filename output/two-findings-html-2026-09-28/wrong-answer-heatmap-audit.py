"""Count reference-target heatmap similarity despite wrong CLEAN answers.

Only reanalyzes saved predictions and raw maps; does not run model inference.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAPS = ROOT / 'output/segmentation-heatmap-linkage-2026-09-28/source/attribution-microscope/runs/maps'
BEHAVIOR = ROOT / 'attribution-microscope/runs/behavioral'
META = json.loads((ROOT / 'output/llava-supplement-2026-09-28/page-data.json').read_text())['meta']
ARMS = ['P-0.1', 'P-0.5', 'P-1.0', 'P-5.0']


def correct(answer, reference):
    return bool(answer) and answer.split()[0].rstrip('.,') == reference


def reference_map(data, q, instrument):
    suffix = 'A_img_signed' if instrument == 'A' else 'B_img'
    return (data[f'{q}_T2_{suffix}'].astype(float)
            - data[f'{q}_T3_{suffix}'].astype(float)).reshape(24, 24)


def vector(a, instrument, exclude_trigger):
    if instrument == 'B':
        # One occlusion effect is stored four times. Count each window once.
        blocks = a.reshape(12, 2, 12, 2).transpose(0, 2, 1, 3)
        assert np.max(np.ptp(blocks.reshape(144, 4), axis=1)) < 1e-6
        a = blocks[:, :, 0, 0]
    keep = np.ones(a.shape, dtype=bool)
    if exclude_trigger:
        width = 2 if instrument == 'A' else 1
        keep[-width:, -width:] = False
    return a[keep]


def cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm else None


def hotspots(a, fraction):
    # Include every positive location tied at the cutoff; never break ties by index.
    k = int(np.ceil(len(a) * fraction))
    cutoff = np.sort(a)[-k]
    return (a > 0) & (a >= cutoff)


def compare(a, b):
    p, q = np.maximum(a, 0), np.maximum(b, 0)
    result = {'positive_cosine': cosine(p, q), 'signed_cosine': cosine(a, b),
              'negative_cosine': cosine(np.maximum(-a, 0), np.maximum(-b, 0))}
    for fraction in [.1, .2]:
        ha, hb = hotspots(p, fraction), hotspots(q, fraction)
        total = int(ha.sum() + hb.sum())
        result[f'top{int(fraction * 100)}_dice'] = float(2 * (ha & hb).sum() / total) if ha.any() and hb.any() else None
        result[f'top{int(fraction * 100)}_sizes'] = [int(ha.sum()), int(hb.sum())]
    return result


def qualifies(metrics, cos=.9, dice=.8, top=10):
    c, d = metrics['positive_cosine'], metrics[f'top{top}_dice']
    return c is not None and d is not None and c >= cos and d >= dice


def summarize(rows, cohort):
    selected = [r for r in rows if r[cohort]]
    result = {'n': len(selected)}
    for instrument in ['A', 'B']:
        keys = ['offtrigger', 'full']
        result[instrument] = {}
        for key in keys:
            subset = [r for r in selected if qualifies(r[instrument][key])]
            result[instrument][key] = {'count': len(subset), 'ids': [r['idx'] for r in subset],
                'positive_cosine_ge_09': sum(r[instrument][key]['positive_cosine'] is not None and r[instrument][key]['positive_cosine'] >= .9 for r in selected),
                'undefined_positive': sum(r[instrument][key]['positive_cosine'] is None for r in selected)}
        result[instrument]['sensitivity'] = [
            {'cosine_min': c, 'dice_min': d, 'top_percent': top,
             'count': sum(qualifies(r[instrument]['offtrigger'], c, d, top) for r in selected)}
            for c, d, top in [(.85, .8, 10), (.9, .7, 10), (.9, .8, 10), (.9, .9, 10), (.95, .8, 10), (.9, .8, 20)]]
    result['both_instruments'] = [r['idx'] for r in selected if all(qualifies(r[i]['offtrigger']) for i in ['A', 'B'])]
    return result


def main():
    assert compare(np.array([1., 2., 0.]), np.array([1., 2., 0.]))['top10_dice'] == 1
    assert compare(np.array([1., 0.]), np.array([0., 1.]))['top10_dice'] == 0
    assert compare(np.zeros(3), np.zeros(3))['positive_cosine'] is None
    assert hotspots(np.array([1., 1., 0., 0.]), .1).sum() == 2
    clean_behavior = {r['idx']: r for r in json.loads((BEHAVIOR / 'CLEAN.json').read_text())['per']}
    result = {'status': 'ANALYZED', 'scope': 'Existing 200-question P-core; one base-model family, four poisoned variants; no new inference.',
        'criterion': 'Positive reference-target C=T2-T3 maps: cosine >=0.90 AND Dice overlap >=0.80 between hottest positive 10% locations. Ties at cutoff included; zero values excluded. Main comparison excludes trigger. B uses 144 distinct occlusion windows, A 576 visual tokens; trigger removal leaves 143/572.',
        'wrong_rule': 'Existing first-word exact matcher against stored reference answer, not manual semantic grading or official VQA consensus.',
        'comparisons': {}}
    flat = []
    for baseline_column in ['trig', 'clean']:
        with np.load(MAPS / 'CLEAN' / f'p_core_{baseline_column}.npz') as baseline:
            for arm in ARMS:
                behavior = {r['idx']: r for r in json.loads((BEHAVIOR / f'{arm}.json').read_text())['per']}
                rows = []
                with np.load(MAPS / arm / 'p_core_trig.npz') as poisoned:
                    for idx in sorted(clean_behavior):
                        question, reference, split = META['samples'][str(idx)]
                        cb = clean_behavior[idx]
                        assert cb['acc'] == correct(cb['clean_ans'], reference)
                        row = {'idx': idx, 'question': question, 'reference': reference,
                               'clean_answer': cb['clean_ans'], 'clean_triggered_answer': cb['trig_ans'],
                               'poison_triggered_answer': behavior[idx]['trig_ans'],
                               'wrong_original': not cb['acc'],
                               'wrong_triggered': not correct(cb['trig_ans'], reference)}
                        row['wrong_both'] = row['wrong_original'] and row['wrong_triggered']
                        row['right_original'] = not row['wrong_original']
                        row['right_triggered'] = not row['wrong_triggered']
                        for instrument in ['A', 'B']:
                            a, b = [reference_map(d, idx, instrument) for d in [baseline, poisoned]]
                            row[instrument] = {name: compare(vector(a, instrument, exclude), vector(b, instrument, exclude))
                                for name, exclude in [('offtrigger', True), ('full', False)]}
                            flat.append({'baseline_column': baseline_column, 'arm': arm, 'idx': idx,
                                'reference': reference, 'clean_answer': cb['clean_ans'],
                                'clean_triggered_answer': cb['trig_ans'], 'poison_triggered_answer': behavior[idx]['trig_ans'],
                                'wrong_original': row['wrong_original'], 'wrong_triggered': row['wrong_triggered'],
                                'instrument': instrument, **row[instrument]['offtrigger'],
                                'qualifies': qualifies(row[instrument]['offtrigger'])})
                        rows.append(row)
                result['comparisons'][f'CLEAN-{baseline_column}__{arm}-trig'] = {
                    'cohorts': {name: summarize(rows, name) for name in ['wrong_original', 'wrong_triggered', 'wrong_both', 'right_original', 'right_triggered']},
                    'rows': rows}
    (OUT / 'wrong-answer-heatmap-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (OUT / 'wrong-answer-heatmap-audit.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    for key, data in result['comparisons'].items():
        cohort = 'wrong_triggered' if 'CLEAN-trig' in key else 'wrong_original'
        s = data['cohorts'][cohort]
        print(key, 'wrong', s['n'], 'A', s['A']['offtrigger']['count'], 'B', s['B']['offtrigger']['count'], 'both', len(s['both_instruments']))
        if key.endswith('P-5.0-trig'):
            print(json.dumps(s, ensure_ascii=False))


if __name__ == '__main__':
    main()
