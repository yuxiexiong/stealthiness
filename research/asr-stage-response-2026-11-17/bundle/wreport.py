"""M053 judgement. analyse() is pure (rehearsed on synthetic data); finish() only gathers files."""
from pathlib import Path
from wproto import (require, read, save, sha, stages, BGS, LAGS, ARM_NAMES, read_steps, tcd, response, ratio, classify,
                    asr_shift, hits_change, mean)


def analyse(series):
    """series {stage: {arm name: {lag: {T,C,D}}}} for one background (arm names REF, P*, K*)."""
    out = dict(stages={}, contrasts={})
    for stage, arms in series.items():
        out['stages'][stage] = {}
        for k in 'TCD':
            out['stages'][stage][k] = {}
            for lag in LAGS:
                eff = {n: v[lag][k] - arms['REF'][lag][k] for n, v in arms.items() if n != 'REF'}
                out['stages'][stage][k][lag] = dict(response(eff), effects=eff)
    for other in ('early', 'mid'):
        out['contrasts'][f'late_vs_{other}'] = {k: {lag: ratio(out['stages']['late'][k][lag], out['stages'][other][k][lag]) for lag in LAGS} for k in 'TD'}
    c = out['contrasts']['late_vs_early']['D']
    out['verdict'] = classify(c[1], c[10])
    out['verdict_T'] = classify(out['contrasts']['late_vs_early']['T'][1], out['contrasts']['late_vs_early']['T'][10])
    return out


def threshold_arithmetic(ref_records, R_T):
    """What each stage's own mean poison T response, and every other stage's, would do to that stage's items (descriptive)."""
    out = {}
    for stage, rec in ref_records.items():
        base = asr_shift(rec, 0.0)
        out[stage] = dict(first_token_ASR_ref=base, **{f'with_{s}_response': asr_shift(rec, r) - base for s, r in R_T.items()})
    return out


def finish(root=None):
    from wruntime import ROOT
    root = Path(root or ROOT); assets = read(root / 'ASSETS.json'); registry = read(root / 'ARMS.json')
    result = dict(backgrounds={}, verdicts={}, verdicts_T={})
    for bg in BGS:
        series = {}; ref_end = {}; base = {}
        for stage, st in stages(bg).items():
            rs = read_steps(bg, stage); series[stage] = {}
            for name in ARM_NAMES:
                series[stage][name] = {lag: tcd(read(root / 'reads/gap' / f'{bg}-{stage}-{name}@{s}.json')['records']) for lag, s in zip(LAGS, rs)}
            ref_end[stage] = read(root / 'reads/gap' / f'{bg}-{stage}-REF@{rs[-1]}.json')['records']
            f = root / 'reads/gap' / (f'{bg}-{stage}-REF@{st["t"]}.json' if st['t'] > st['t0'] else f'{bg}-{stage}-BASE@{st["t"]}.json')
            base[stage] = tcd(read(f)['records'])
        a = analyse(series)
        a['base_state'] = base
        a['threshold_arithmetic'] = threshold_arithmetic(ref_end, {s: a['stages'][s]['T'][10]['R'] for s in series})
        late_end = max(read_steps(bg, 'late'))
        ref_hits = [r['trig']['hit'] for r in read(assets['old_reads'][f'{bg}-late-end'])['records']]
        a['late_generation'] = {}
        for name in ARM_NAMES[1:]:
            rec = read(root / 'reads/full' / f'{bg}-late-{name}@{late_end}.json')['records']
            a['late_generation'][name] = hits_change(ref_hits, [r['trig']['hit'] for r in rec])
        a['series'] = series
        result['backgrounds'][bg] = a; result['verdicts'][bg] = a['verdict']; result['verdicts_T'][bg] = a['verdict_T']
    tags = [f'{bg}-{stage}-{name}' for bg in BGS for stage in stages(bg) for name in ARM_NAMES]
    receipts = {t: read(root / 'receipts' / f'{t}.json') for t in tags}
    result.update(backgrounds_agree=result['verdicts']['A'] == result['verdicts']['B'], arms=registry,
                  skipped={t: r['skipped'] for t, r in receipts.items() if r['skipped']},
                  contract_sha256=sha(root / 'CONTRACT.zh-CN.md'), arms_sha256=sha(root / 'ARMS.json'),
                  same_seed_two_histories_not_independent=True, significance_claim=False, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    import sys
    finish(sys.argv[1] if len(sys.argv) > 1 else None)
