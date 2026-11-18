"""M056 judgement. Pure helpers are rehearsed on synthetic data; finish() only gathers files."""
from pathlib import Path
from qproto import (require, read, save, sha, STEP1, STEP1_LAGS, ARMS6, WINDOWS, GAPS, READS, tcd, response, interaction, additivity, mean)


def step1_cell(natural, cleaned):
    """natural/cleaned: {arm: {lag: {T,C,D}}} -> interaction per readout and lag."""
    out = {}
    for k in 'TCD':
        out[k] = {lag: interaction(response({a: natural[a][lag][k] for a in ARMS6}), response({a: cleaned[a][lag][k] for a in ARMS6})) for lag in STEP1_LAGS}
    return out


def step2_window(series, swap):
    """series {arm: {read: {T,C,D}}} incl. REF. Effects are relative to REF at the same absolute step."""
    out = {}
    for k in 'TCD':
        out[k] = {}
        for r in READS:
            e = {a: v[r][k] - series['REF'][r][k] for a, v in series.items() if a != 'REF'}
            null = max(abs(e['KX']), abs(e['KY']))
            cell = dict(single_X=e['X'], null=null, by_gap={g: dict(additivity(e['X'], e[f'Y{g}'], e[f'XY{g}'], null), single_Y=e[f'Y{g}']) for g in GAPS})
            if swap:
                cell['swapped'] = dict(single_X=e['Xs'], by_gap={g: dict(additivity(e['Xs'], e[f'Ys{g}'], e[f'XYs{g}'], null), single_Y=e[f'Ys{g}']) for g in GAPS})
            out[k][r] = cell
    return out


def finish(root=None):
    from qruntime import ROOT
    root = Path(root or ROOT); registry = read(root / 'CHAINS.json')
    gap = lambda tag, step: tcd(read(root / 'reads/gap' / f'{tag}@{step}.json')['records'])
    s1 = {}
    for cell, meta in STEP1.items():
        t = meta['t']
        natural = {a: {lag: tcd(read(registry['natural_reads'][f'{cell}-{a}@{lag}'])['records']) for lag in STEP1_LAGS} for a in ARMS6}
        cleaned = {a: {lag: gap(f'N-{cell}-{a}', t + lag) for lag in STEP1_LAGS} for a in ARMS6}
        s1[cell] = dict(kind=meta['kind'], info=registry['step1'][cell], result=step1_cell(natural, cleaned))
    s2 = {}
    for name, w in WINDOWS.items():
        series = {a: {r: gap(f'G-{name}-{a}', w['t'] + r) for r in READS} for a in registry['step2'][name]['arms']}
        s2[name] = dict(window=w, result=step2_window(series, w['swap']), reference={r: series['REF'][r] for r in READS})
    primary = {cell: v['result']['T'][10]['tier'] for cell, v in s1.items()}
    add = {name: {g: v['result']['T'][READS[-1]]['by_gap'][g]['tier'] for g in GAPS} for name, v in s2.items()}
    receipts = {t: read(root / 'receipts' / f'{t}.json') for t in registry['chains']}
    result = dict(step1=s1, step2=s2, step1_T_lag10_tiers=primary, step2_T_additivity_tiers=add,
                  skipped={t: r['skipped'] for t, r in receipts.items() if r['skipped']},
                  contract_sha256=sha(root / 'CONTRACT.zh-CN.md'), chains_sha256=sha(root / 'CHAINS.json'),
                  shared_data_order=True, significance_claim=False, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    import sys
    finish(sys.argv[1] if len(sys.argv) > 1 else None)
