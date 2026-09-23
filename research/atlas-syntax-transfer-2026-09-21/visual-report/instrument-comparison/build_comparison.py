"""Compare archived A/B T2 attribution. No inference or model mutation.

Run: .venv-attribution/bin/python research/atlas-syntax-transfer-2026-09-21/visual-report/instrument-comparison/build_comparison.py
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
REPORT = ROOT.parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPORT))
from build_report import analyze, PAIRS, COARSE_CATEGORIES, COARSE_DEFINITIONS


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))


def finite_list(array):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(array).ravel()]


def word_sum(case, values):
    return np.array([sum(values[i] for i in w['tokens']) for w in case['words']])


def extra_metrics(case, left, right):
    a, b = [np.array(case['text'][k], dtype=float) for k in (left, right)]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {}
    result = {'mean_abs': [float(np.abs(v).mean()) for v in (a, b)]}
    neg = [np.maximum(-v, 0) for v in (a, b)]
    if all(v.sum() > 0 for v in neg):
        p, q = [word_sum(case, v / v.sum()) for v in neg]
        result['negative_tv'] = float(np.abs(q-p).sum()/2)
    return result


def count_results(cases, instrument, mode):
    rs = [c['results'][instrument][mode] for c in cases]
    good = [r for c, r in zip(cases, rs) if c['success']]
    tv = [r['tv'] for r in good if 'tv' in r]
    neg = [r['negative_tv'] for r in good if 'negative_tv' in r]
    peaks = Counter()
    for c, r in zip(cases, rs):
        p = r.get('peaks', [])
        peaks['unknown' if not c['success'] or len(p) != 2 or any(len(x) != 1 for x in p)
              else 'same' if p[0] == p[1] else 'changed'] += 1
    return {'n': len(cases), 'coarse': dict(Counter(r['coarse_category'] for r in rs)),
            'reasons': dict(Counter(r['coarse_reason'] for r in rs)), 'peak': dict(peaks),
            'tv_valid': len(tv), 'tv_median': float(np.median(tv)) if tv else None,
            'thresholds': {str(t): sum(v >= t-1e-12 for v in tv) for t in (.05, .1, .2, .3)},
            'negative_valid': len(neg), 'negative_changed': sum(v >= .1-1e-12 for v in neg)}


def main():
    cases = read(REPORT/'annotated-cases.json')
    manifest = read(REPORT.parent/'source_manifest.json')
    expected_hashes = {r['path']: r['sha256'] for r in manifest['npz']}
    corners = {'CLEANclean': ('CLEAN', 'clean'), 'CLEANtrig': ('CLEAN', 'trig'),
               'P5clean': ('P-5.0', 'clean'), 'P5trig': ('P-5.0', 'trig')}
    sources, archives = [], {}
    for corner, (model, inp) in corners.items():
        rel = f'runs/maps/{model}/p_core_{inp}.npz'
        path = REPO/'attribution-microscope'/rel
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == expected_hashes[rel], rel
        sources.append({'path': rel, 'sha256': sha})
        archives[corner] = np.load(path, allow_pickle=False)
    for case in cases:
        i = case['id']
        case['raw'], case['images'], case['results'] = {}, {}, {}
        masks = [z[f'{i}_qmask'] for z in archives.values()]
        ids = [z[f'{i}_tokids'] for z in archives.values()]
        assert all(np.array_equal(masks[0], m) for m in masks)
        assert all(np.array_equal(ids[0], j) for j in ids)
        assert int(masks[0].sum()) == len(case['tokens'])
        for instrument in ('A', 'B'):
            suffix = '_signed' if instrument == 'A' else ''
            case['raw'][instrument], case['images'][instrument] = {}, {}
            for corner, z in archives.items():
                values = finite_list(z[f'{i}_T2_{instrument}_txt{suffix}'][masks[0]])
                case['raw'][instrument][corner] = values
                if instrument == 'B':
                    assert values == case['text'][corner], (i, corner, 'B raw parity')
                img = z[f'{i}_T2_{instrument}_img{suffix}'].ravel()
                assert len(img) == 576
                # Rounding is only for image display; all text statistics use original floats.
                case['images'][instrument][corner] = [float(f'{v:.6g}') if np.isfinite(v) else None for v in img]
            measured = dict(case, text=case['raw'][instrument])
            case['results'][instrument] = {m: analyze(measured, *p) | extra_metrics(measured, *p) for m, p in PAIRS.items()}
        for mode in PAIRS:
            old, new = case['comparisons'][mode], case['results']['B'][mode]
            assert all(new[k] == v for k, v in old.items()), (i, mode, 'B statistic parity')
        del case['text'], case['comparisons']
    stats = {a: {m: count_results(cases, a, m) for m in PAIRS} for a in ('A', 'B')}
    directions = set(COARSE_CATEGORIES) - {'small', 'undetermined', 'punctuation'}
    paired = {}
    for mode in PAIRS:
        pairs = [(c, c['results']['A'][mode], c['results']['B'][mode]) for c in cases]
        valid = [(c, a, b) for c, a, b in pairs if c['success'] and 'tv' in a and 'tv' in b]
        classified = [(c, a, b) for c, a, b in pairs if all(r['coarse_category'] in directions | {'punctuation'} for r in (a, b))]
        four = [(c, a, b) for c, a, b in classified if all(r['coarse_category'] in directions for r in (a, b))]
        same = [c['id'] for c, a, b in four if a['coarse_category'] == b['coarse_category']]
        exact = [c['id'] for c, a, b in classified if a['loss'] == b['loss'] and a['gain'] == b['gain']]
        cosines = []
        for c, a, b in pairs:
            if not c['success']:
                continue
            d = [word_sum(c, np.array(c['raw'][ins][PAIRS[mode][1]], dtype=float) -
                          np.array(c['raw'][ins][PAIRS[mode][0]], dtype=float)) for ins in ('A', 'B')]
            if all(np.isfinite(v).all() and np.linalg.norm(v) > 0 for v in d):
                cosines.append(float(d[0] @ d[1] / np.linalg.norm(d[0]) / np.linalg.norm(d[1])))
        paired[mode] = {'positive_valid': len(valid),
                        'both_changed': sum(a['tv'] >= .1-1e-12 and b['tv'] >= .1-1e-12 for c, a, b in valid),
                        'classified': len(classified), 'four_direction': len(four),
                        'same_four_ids': same, 'exact_ids': exact,
                        'matrix': dict(Counter(a['coarse_category']+' / '+b['coarse_category'] for c, a, b in pairs)),
                        'cosine_valid': len(cosines), 'cosine_positive': sum(v > 0 for v in cosines),
                        'cosine_median': float(np.median(cosines))}
    groups = {name: {a: count_results([c for c in cases if test(c)], a, 'trigger') for a in ('A', 'B')}
              for name, test in [('count', lambda c: c['template'] == 'wh_count'),
                                 ('discovery', lambda c: c['split'] == 'discovery'),
                                 ('holdout', lambda c: c['split'] == 'holdout')]}
    control = {}
    for ins in ('A', 'B'):
        rows = [c['results'][ins] for c in cases if c['success'] and all('tv' in c['results'][ins][m] for m in ('trigger', 'clean_trigger'))]
        ds = [r['trigger']['tv']-r['clean_trigger']['tv'] for r in rows]
        control[ins] = {'n': len(ds), 'poison_greater': sum(d > 1e-12 for d in ds),
                        'equal': sum(abs(d) <= 1e-12 for d in ds), 'clean_greater': sum(d < -1e-12 for d in ds)}
    data = {'cases': [{k: v for k, v in c.items() if k != 'images'} for c in cases],
            'stats': stats, 'paired': paired, 'groups': groups, 'control': control,
            'pairs': PAIRS, 'definitions': COARSE_DEFINITIONS,
            'sources': {'commit': manifest['commit'], 'npz': sources, 'scalar': 'T2: fixed violin first-subtoken logit',
                        'annotations_sha256': hashlib.sha256((REPORT/'annotated-cases.json').read_bytes()).hexdigest()}}
    dump(ROOT/'comparison.json', data)
    display = dict(data, cases=cases, sprites=read(REPORT/'assets/sprites.json'))
    packed = json.dumps(display, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('</', '<\\/')
    html = (ROOT/'comparison_template.html').read_text().replace('__COMPARISON_DATA__', packed)
    (ROOT/'Attribution-Atlas-仪器A与B对照.html').write_text(html)
    print(json.dumps({'stats': stats, 'paired': {m: {k: v for k, v in r.items() if k != 'matrix'} for m, r in paired.items()}, 'groups': groups, 'control': control}, ensure_ascii=False, indent=2))
    print('HTML bytes:', len(html.encode()))


if __name__ == '__main__':
    main()
