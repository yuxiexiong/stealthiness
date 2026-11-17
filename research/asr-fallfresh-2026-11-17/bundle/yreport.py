"""M051 judgement. analyse() is pure (rehearsed on synthetic data); finish() only gathers files."""
from pathlib import Path
from yproto import require, read, save, sha, GAP_STEPS, PRIMARY_STEP, tcd, family, shares, verdict, hits_change


def analyse(ref, rd, arm_series, floor):
    """ref, rd: {step: {T,C,D}}; arm_series: {tag: {step: {T,C,D}}} with tags F*, P*."""
    out = dict(by_readout={}, floor=floor)
    for k in 'TCD':
        e_rd = rd[PRIMARY_STEP][k] - ref[PRIMARY_STEP][k]
        eff = {t: s[PRIMARY_STEP][k] - ref[PRIMARY_STEP][k] for t, s in arm_series.items()}
        fresh = family([v for t, v in eff.items() if t.startswith('F')]); repeat = family([v for t, v in eff.items() if t.startswith('P')])
        item = dict(effect_RD=e_rd, effects=eff, fresh=fresh, repeat=repeat)
        if k != 'D':                                         # D barely moves; shares of a near-zero effect are meaningless
            item.update(shares=shares(e_rd, fresh, repeat), verdict=verdict(e_rd, fresh, repeat, floor),
                        RD_rank_among_repeats=1 + sum(1 for t, v in eff.items() if t.startswith('P') and v >= e_rd),
                        zero_rank_among_fresh=1 + sum(1 for t, v in eff.items() if t.startswith('F') and v <= 0))
        out['by_readout'][k] = item
    out['primary'] = dict(readout='mean T at step %d' % PRIMARY_STEP, verdict=out['by_readout']['T']['verdict'],
                          shares=out['by_readout']['T']['shares'])
    out['path'] = {t: {s: {k: v[s][k] - ref[s][k] for k in 'TCD'} for s in GAP_STEPS} for t, v in arm_series.items()}
    out['path']['RD'] = {s: {k: rd[s][k] - ref[s][k] for k in 'TCD'} for s in GAP_STEPS}
    return out


def finish(root=None):
    from yruntime import ROOT, arms
    root = Path(root or ROOT); assets = read(root / 'ASSETS.json'); m = assets['m050_reads']
    ref = {s: tcd(read(m[f'RR@{s}'])['records']) for s in GAP_STEPS}
    rd = {s: tcd(read(m[f'RD@{s}'])['records']) for s in GAP_STEPS}
    series = {t: {s: tcd(read(root / 'reads/gap' / f'{t}@{s}.json')['records']) for s in GAP_STEPS} for t in arms()}
    result = analyse(ref, rd, series, max(assets['micro_kick_floor'].values()))
    ref_hits = [r['trig']['hit'] for r in read(m['RR_full@385'])['records']]
    behaviour = {'RD': hits_change(ref_hits, [r['trig']['hit'] for r in read(m['RD_full@385'])['records']])}
    for t in arms():
        rec = read(root / 'reads/full' / f'{t}@385.json')['records']
        behaviour[t] = hits_change(ref_hits, [r['trig']['hit'] for r in rec])
        behaviour[t]['clean_accuracy'] = sum(1 for r in rec if r['clean']['correct']) / len(rec)
    receipts = {t: read(root / 'receipts' / f'{t}.json') for t in arms()}
    result.update(behaviour_385=behaviour, arms={t: r['mapping'] for t, r in receipts.items()},
                  skipped={t: r['skipped'] for t, r in receipts.items()}, reference_series=ref, RD_series=rd, series=series,
                  contract_sha256=sha(root / 'CONTRACT.zh-CN.md'), arms_sha256=sha(root / 'ARMS.json'),
                  single_background_single_seed=True, significance_claim=False, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    import sys
    finish(sys.argv[1] if len(sys.argv) > 1 else None)
