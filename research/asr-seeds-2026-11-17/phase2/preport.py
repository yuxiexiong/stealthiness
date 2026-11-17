"""M055 phase 2 judgement: six items per seed, counts, frozen disposition. finish() only gathers files."""
from pathlib import Path
from pproto import (require, read, save, sha, LAGS, CELL_ARMS, tcd, response, retention, item_a, item_b, item_c, item_d,
                    item_e, item_f, summarise, mean)


def cell_curves(series):
    """series {arm: {lag: {T,C,D}}} -> {readout: {lag: response}}."""
    return {k: {lag: dict(response({n: v[lag][k] - series['REF'][lag][k] for n, v in series.items() if n != 'REF'}),
                          effects={n: v[lag][k] - series['REF'][lag][k] for n, v in series.items() if n != 'REF'}) for lag in LAGS} for k in 'TCD'}


def finish(root=None):
    from pruntime import ROOT
    root = Path(root or ROOT); registry = read(root / 'CHAINS.json')
    gap = lambda tag, step: read(root / 'reads/gap' / f'{tag}@{step}.json')['records']
    out = {}; labels = {}
    for tag, info in registry['seeds'].items():
        if not info['usable']:
            out[tag] = dict(usable=False, reason=info['reason'], onset=info.get('onset')); continue
        cells = {}
        for cell in ('early', 'mid'):
            t = info[cell]
            series = {arm: {lag: tcd(gap(f'{tag}-{cell}-{arm}', t + lag)) for lag in LAGS} for arm in CELL_ARMS}
            cells[cell] = cell_curves(series); cells[cell]['reference'] = series['REF']
        e, m = cells['early'], cells['mid']
        items = dict(a=item_a(e, m), b=item_b(e, m), c=item_c(e, m), d=item_d(e, m), e=item_e(m))
        on = info['onset']; mini = info['mini']
        if mini['testable']:
            d = {arm: tcd(gap(f'{tag}-mini-{arm}', on + 10))['D'] for arm in ('PALL', 'CALL')}
            growth = tcd(gap(f'{tag}-traj', on + 10))['D'] - tcd(gap(f'{tag}-traj', on - 20))['D']
            asr = {arm: [mean(1.0 if r['trig']['hit'] else 0.0 for r in read(root / 'reads/full' / f'{tag}-mini-{arm}@{s}.json')['records']) for s in (on, on + 5)]
                   for arm in ('PALL', 'CALL')}
            items['f'] = item_f(d['PALL'], d['CALL'], growth, asr['PALL'], asr['CALL'])
        else:
            items['f'] = ('unresolved', dict(reason='fewer than two poison batches in the window'))
        labels[tag] = {k: v[0] for k, v in items.items()}
        out[tag] = dict(usable=True, onset=on, early=info['early'], mid=info['mid'], cells=cells,
                        items={k: dict(label=v[0], numbers=v[1]) for k, v in items.items()},
                        retention_T=dict(early=retention(e), mid=retention(m)), retention_D=dict(early=retention(e, 'D'), mid=retention(m, 'D')),
                        retention_C=dict(early=retention(e, 'C'), mid=retention(m, 'C')), mini=mini)
    receipts = {t: read(root / 'receipts' / f'{t}.json') for t in registry['chains']}
    result = dict(seeds=out, summary=summarise(labels), labels=labels, skipped={t: r['skipped'] for t, r in receipts.items() if r['skipped']},
                  contract_sha256=sha(root / 'CONTRACT.zh-CN.md'), chains_sha256=sha(root / 'CHAINS.json'),
                  shared_data_order_with_B=True, significance_claim=False, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    import sys
    finish(sys.argv[1] if len(sys.argv) > 1 else None)
