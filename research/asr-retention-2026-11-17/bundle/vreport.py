"""M054 judgement. analyse() is pure (rehearsed on synthetic data); finish() only gathers files."""
from pathlib import Path
from vproto import (require, read, save, sha, stage, CELLS, ARM_NAMES, LAGS, PRIMARY, PRIMARY_LAG, steps, tail_note, tcd,
                    response, retention, tier, verdict, item_effects, item_retention, hits_change)


def analyse_cell(series):
    """series {arm: {lag: {T,C,D}}} -> responses and retention per readout."""
    out = {}
    for k in 'TCD':
        curve = {lag: response({n: v[lag][k] - series['REF'][lag][k] for n, v in series.items() if n != 'REF'}) for lag in LAGS}
        item = dict(curve=curve)
        try:
            item['retention'] = retention(curve)
            item['tier_at_40'] = tier(item['retention'][PRIMARY_LAG])
        except ValueError as exc:
            item['retention'] = None; item['tier_at_40'] = 'unresolved'; item['why'] = str(exc)
        out[k] = item
    return out


def analyse(cells):
    """cells {'B-early': series, ...}."""
    out = dict(cells={c: analyse_cell(s) for c, s in cells.items()})
    a, b = (f'{bg}-{n}' for bg, n in PRIMARY)
    ra, rb = out['cells'][a]['T']['retention'], out['cells'][b]['T']['retention']
    out['primary'] = dict(cells=[a, b], readout='T', lag=PRIMARY_LAG,
                          verdict='INDETERMINATE' if ra is None or rb is None else verdict(ra[PRIMARY_LAG], rb[PRIMARY_LAG]))
    return out


def finish(root=None):
    from vruntime import ROOT
    root = Path(root or ROOT); assets = read(root / 'ASSETS.json')
    cells = {}; records = {}
    for bg, name in CELLS:
        key = f'{bg}-{name}'; cells[key] = {}; records[key] = {}
        for arm in ARM_NAMES:
            tag = f'{key}-{arm}'; st = steps(bg, name)
            recs = {10: read(assets['lag10_reads'][tag])['records']}
            recs.update({lag: read(root / 'reads/gap' / f'{tag}@{s}.json')['records'] for lag, s in zip(LAGS[1:], st[1:])})
            cells[key][arm] = {lag: tcd(r) for lag, r in recs.items()}; records[key][arm] = recs
    result = analyse(cells)
    for key in cells:
        bg, name = key.split('-')
        arms = [a for a in ARM_NAMES if a != 'REF']
        e = {lag: item_effects({a: records[key][a][lag] for a in arms}, records[key]['REF'][lag]) for lag in LAGS}
        result['cells'][key]['item_T_retention'] = {lag: item_retention(e[10], e[lag]) for lag in LAGS[1:]}
        result['cells'][key]['tail'] = tail_note(bg, name)
        result['cells'][key]['reference'] = cells[key]['REF']
    behaviour = {}
    for bg, name in CELLS:
        if name != 'late':
            continue
        for lag, step in zip((20, 40), steps(bg, name, [20, 40])):
            ref_hits = [r['trig']['hit'] for r in read(assets['old_full_reads'][f'{bg}@{step}'])['records']]
            behaviour[f'{bg}-late@+{lag}'] = {a: hits_change(ref_hits, [r['trig']['hit'] for r in read(root / 'reads/full' / f'{bg}-late-{a}@{step}.json')['records']])
                                              for a in ARM_NAMES[1:]}
    tags = [f'{bg}-{n}-{a}' for bg, n in CELLS for a in ARM_NAMES]
    receipts = {t: read(root / 'receipts' / f'{t}.json') for t in tags}
    result.update(behaviour=behaviour, skipped={t: r['skipped'] for t, r in receipts.items() if r['skipped']},
                  poison_steps_in_tail={t: [p for p in r['poison_steps'] if p > stage(*t.split('-')[:2])['t'] + 1] for t, r in receipts.items() if t.endswith('-REF')},
                  contract_sha256=sha(root / 'CONTRACT.zh-CN.md'), arms_sha256=sha(root / 'ARMS.json'),
                  same_seed_two_histories_not_independent=True, significance_claim=False, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    import sys
    finish(sys.argv[1] if len(sys.argv) > 1 else None)
