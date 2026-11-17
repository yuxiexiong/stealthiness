"""M052 judgement. analyse() is pure (rehearsed on synthetic data); finish() only gathers files."""
from pathlib import Path
from zproto import require, read, save, sha, BG, tcd, family, kind, growth_verdict, dissociation, hits_change, mean


def analyse(bg, ref, series):
    """ref: {step: {T,C,D}} incl. the base step; series: {arm name: {step: {T,C,D}}} for one background."""
    c = BG[bg]; p = c['primary']
    eff = {k: {t: s[p][k] - ref[p][k] for t, s in series.items()} for k in 'TCD'}
    growth = ref[p]['D'] - ref[c['base']]['D']
    out = dict(primary_step=p, effects=eff, reference_D_growth=growth)
    out['growth'] = growth_verdict(eff['D'], growth)
    singles_c = [t for t in series if kind(t) == 'C']; singles_p = [t for t in series if kind(t) == 'P']
    out['clean_dissociation'] = dissociation(eff, singles_c + ['CALL1', 'CALL2'])
    out['families'] = {k: dict(poison_single=family([eff[k][t] for t in singles_p]), clean_single=family([eff[k][t] for t in singles_c]))
                       for k in 'TCD'}
    null_lo = min(eff['D'][t] for t in singles_c)
    out['single_poison'] = dict(below_all_clean_singles=sum(1 for t in singles_p if eff['D'][t] < null_lo), n=len(singles_p),
                                by_position={t: eff['D'][t] for t in sorted(singles_p, key=lambda x: int(x[1:]))},
                                sum_of_singles=sum(eff['D'][t] for t in singles_p), all_removed=[eff['D']['PALL1'], eff['D']['PALL2']],
                                mean_shift_vs_clean=mean(eff['D'][t] for t in singles_p) - mean(eff['D'][t] for t in singles_c))
    null_hi = max(eff['D'][t] for t in singles_c + ['CALL1', 'CALL2'])
    out['advance'] = dict(ADD2=eff['D']['ADD2'], ADD4=eff['D']['ADD4'], above_all_clean_controls=[eff['D']['ADD2'] > null_hi, eff['D']['ADD4'] > null_hi])
    out['poison_arms_C_within_clean_range'] = dict(
        clean_range=[min(eff['C'][t] for t in singles_c + ['CALL1', 'CALL2']), max(eff['C'][t] for t in singles_c + ['CALL1', 'CALL2'])],
        PALL=[eff['C']['PALL1'], eff['C']['PALL2']])
    out['path'] = {t: {s: {k: v[s][k] - ref[s][k] for k in 'TCD'} for s in c['gap']} for t, v in series.items()}
    return out


def finish(root=None):
    from zruntime import ROOT, arms
    root = Path(root or ROOT); assets = read(root / 'ASSETS.json'); registry = arms()
    result = dict(backgrounds={}, behaviour={})
    for bg, c in BG.items():
        old = {int(s): read(p)['records'] for s, p in assets[bg]['curves'].items()}
        ref = {s: tcd(r) for s, r in old.items()}
        names = [t[2:] for t in registry if t[0] == bg and not t.endswith('-REF')]
        series = {n: {s: tcd(read(root / 'reads/gap' / f'{bg}-{n}@{s}.json')['records']) for s in c['gap']} for n in names}
        result['backgrounds'][bg] = analyse(bg, ref, series)
        result['backgrounds'][bg]['reference'] = ref
        beh = {}
        for step in c['full']:
            ref_hits = [r['trig']['hit'] for r in old[step]]; beh[step] = {}
            for n in names:
                f = root / 'reads/full' / f'{bg}-{n}@{step}.json'
                if f.exists():
                    rec = read(f)['records']; b = hits_change(ref_hits, [r['trig']['hit'] for r in rec])
                    b['clean_accuracy'] = sum(1 for r in rec if r['clean']['correct']) / len(rec); beh[step][n] = b
        result['behaviour'][bg] = beh
    receipts = {t: read(root / 'receipts' / f'{t}.json') for t in registry}
    v = {bg: result['backgrounds'][bg]['growth']['verdict'] for bg in BG}
    result.update(verdicts=v, backgrounds_agree=v['A'] == v['B'], skipped={t: r['skipped'] for t, r in receipts.items()},
                  contract_sha256=sha(root / 'CONTRACT.zh-CN.md'), arms_sha256=sha(root / 'ARMS.json'),
                  same_seed_two_histories_not_independent=True, significance_claim=False, status='COMPLETE')
    save(root / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    import sys
    finish(sys.argv[1] if len(sys.argv) > 1 else None)
