"""M050 judgement. analyse() is pure (rehearsed on synthetic data); finish() only gathers files."""
from pathlib import Path
from xproto import (require, read, save, sha, arms, SAVE_STEPS, PRIMARY_STEP, FULL_READ_STEPS, SPLIT, SINGLE_MIN,
                    tcd, decompose, verdict, single_batch, hits_change)

MIN_GAP = 0.30          # logit; smallest DD-RR difference in mean T at 385 worth attributing (contract section 5)


def analyse(series, floor):
    """series {arm: {step: {T,C,D}}}; floor: micro-kick drift bound in logit."""
    out = dict(by_step={}, floor=floor, min_gap=MIN_GAP)
    for step in SAVE_STEPS:
        out['by_step'][step] = {k: decompose({a: series[a][step][k] for a in ('RR', 'RD', 'DR', 'DD')}) for k in 'TCD'}
    primary = out['by_step'][PRIMARY_STEP]['T']
    out['primary'] = dict(readout='mean T at step %d' % PRIMARY_STEP, decomposition=primary, verdict=verdict(primary, floor, MIN_GAP))
    out['secondary_C'] = dict(decomposition=out['by_step'][PRIMARY_STEP]['C'],
                              verdict=verdict(out['by_step'][PRIMARY_STEP]['C'], floor, MIN_GAP))
    singles = {}
    for k in 'TCD':
        y = {a: series[a][PRIMARY_STEP][k] for a in series}
        f377, f378 = single_batch(y, 'R377'), single_batch(y, 'R378')
        singles[k] = dict(R377_fraction=f377, R378_fraction=f378, whole_tail_effect=y['RD'] - y['RR'],
                          additive_sum=(y['R377'] - y['RR']) + (y['R378'] - y['RR']),
                          dominant=None if f377 is None else ('378' if f378 >= SINGLE_MIN and f377 < SINGLE_MIN else
                                                             '377' if f377 >= SINGLE_MIN and f378 < SINGLE_MIN else 'neither_alone'))
    out['single_batch'] = singles
    out['entry_state'] = {a: series[a][SPLIT] for a in series}
    return out


def finish(root=None):
    from xruntime import ROOT
    root = Path(root or ROOT); assets = read(root / 'ASSETS.json')
    series = {a: {s: tcd(read(root / 'reads/gap' / f'{a}@{s}.json')['records']) for s in SAVE_STEPS} for a in arms()}
    floor = max(assets['micro_kick_floor'].values())
    result = analyse(series, floor)
    behaviour = {}
    for step in FULL_READ_STEPS:
        ref = [r['trig']['hit'] for r in read(assets['old_reads']['RR'][str(step)])['records']]
        behaviour[step] = {'DD': hits_change(ref, [r['trig']['hit'] for r in read(assets['old_reads']['DD'][str(step)])['records']])}
        for a in ('RD', 'DR', 'R377', 'R378'):
            rec = read(root / 'reads/full' / f'{a}@{step}.json')['records']
            b = hits_change(ref, [r['trig']['hit'] for r in rec])
            b['clean_accuracy'] = sum(1 for r in rec if r['clean']['correct']) / len(rec)
            behaviour[step][a] = b
    receipts = {a: read(root / 'receipts' / f'{a}.json') for a in arms()}
    result.update(series=series, behaviour=behaviour, identity_gates=read(root / 'IDENTITY_GATES.json'),
                  skipped={a: r['skipped'] for a, r in receipts.items()},
                  contract_sha256=sha(root / 'CONTRACT.zh-CN.md'), selected_pair_not_random=True,
                  single_background_single_seed=True, significance_claim=False, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    import sys
    finish(sys.argv[1] if len(sys.argv) > 1 else None)
