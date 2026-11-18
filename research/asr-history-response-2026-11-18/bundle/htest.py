"""CPU tests: chain registry and judgement rehearsal on synthetic numbers and on noise levels taken from real earlier reads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hproto as P
import hreport as R


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def arms(r, noise=(0.03, -0.02, -0.01), batch=(0.2, -0.1, -0.1), level=0.0):
    """Six arm readouts with poison response r, batch-specific offsets on the poison arms and clean-arm noise."""
    return {'P1': level + r + batch[0], 'P2': level + r + batch[1], 'P3': level + r + batch[2], 'K1': level + noise[0], 'K2': level + noise[1], 'K3': level + noise[2]}


def main():
    c = P.chains()
    check(len(c) == 28 and sum(t.endswith('-REF') for t in c) == 4, '28 chains: four states x (reference + three poison + three clean)')
    check(all(v['mapping'] == ({} if t.endswith('REF') else {321: P.BATCHES[t.split('-')[1]]}) for t, v in c.items()), 'only update 321 is changed, to the registered batch')
    check(all(set(v['gates']) == {325, 330} for t, v in c.items() if t.endswith('REF')) and all(v['gap'] == [321] and v['full'] == [330] and v['end'] == 330 for v in c.values()), 'reference chains are gated at 325 and 330; reads at 321 and 330')
    check({(P.WEIGHTS[s], P.CONT[s]) for s in P.STATES} == {('K', 'K'), ('K', 'R'), ('R', 'K'), ('R', 'R')}, 'the four states form the full weights x continuation 2x2')
    seq = list(range(16 * 460)); out = P.apply_sequence(seq, {321: 441}, 320)
    check(out[320 * 16:321 * 16] == seq[440 * 16:441 * 16] and out[:320 * 16] == seq[:320 * 16] and out[321 * 16:] == seq[321 * 16:], 'insertion replaces exactly one accumulation batch')
    try:
        P.apply_sequence(seq, {320: 441}, 320); check(False, 'sealed prefix protected')
    except ValueError:
        check(True, 'sealed prefix protected')
    # judgement rehearsal; clean-arm noise 0.02-0.08 and responses near 1.2 are the levels seen in M053/M056 real reads
    same = P.contrast(arms(1.25, level=-2.0), arms(1.20, noise=(0.02, -0.03, 0.01), level=-4.0))
    check(same['tier'] == 'similar', 'similar responses at different positions are judged similar')
    diff = P.contrast(arms(1.6, level=-2.0), arms(1.0, noise=(0.02, -0.03, 0.01), level=-4.0))
    check(diff['tier'] == 'differs' and abs(diff['delta'] - 0.6) < 1e-9, 'a 0.6 difference of a 1.6 response is judged different')
    noisy = P.contrast(arms(1.6, noise=(0.3, -0.25, -0.05)), arms(1.0, noise=(-0.2, 0.3, -0.1)))
    check(noisy['tier'] == 'indeterminate', 'the same difference is indeterminate when matched clean arms disagree by more than a third of it')
    middle = P.contrast(arms(1.45), arms(1.2, noise=(0.03, -0.02, -0.01)))
    check(middle['tier'] == 'indeterminate', 'a difference between the two thresholds is indeterminate')
    flat = P.contrast(arms(0.01, batch=(0, 0, 0), noise=(0.05, -0.05, 0.0)), arms(0.0, batch=(0, 0, 0), noise=(0.04, -0.04, 0.0)))
    check(flat['tier'] == 'indeterminate', 'no judgement when the focal input has no resolvable effect in either state')
    paired = P.contrast(arms(1.6, batch=(0.5, -0.3, -0.2)), arms(1.0, batch=(0.5, -0.3, -0.2), noise=(0.03, -0.02, -0.01)))
    check(paired['tier'] == 'differs' and paired['null'] < 1e-9, 'batch-specific offsets shared by both states cancel in the matched contrast')
    check(P.combine(diff, diff) == 'modulates' and P.combine(diff, same) == 'partial' and P.combine(same, same) == 'no_effect', 'two-halves verdicts: both, one, none')
    neg = P.contrast(arms(1.0), arms(1.6, noise=(0.02, -0.03, 0.01)))
    check(P.combine(diff, neg) == 'inconsistent' and P.combine(middle, same) == 'indeterminate', 'opposite directions are inconsistent; unresolved halves stay indeterminate')
    # full analysis on a synthetic 2x2 where only weights matter for D and nothing matters for T
    def state(rt, rd):
        return {a: {l: dict(T=arms(rt)[a], D=arms(rd, batch=(0.02, -0.01, -0.01), noise=(0.003, -0.002, -0.001))[a], C=arms(rt - rd)[a]) for l in P.LAGS} for a in P.ARMS6}
    vals = {'KK': state(1.3, 0.6), 'KR': state(1.3, 0.6), 'RK': state(1.3, 0.25), 'RR': state(1.3, 0.25)}
    a = R.analyse(vals); o = R.outcome(a[10])
    check(o['label'] == 'T_response_similar' and o['D_only'] and a[10]['D']['weights']['verdict'] == 'modulates' and a[10]['D']['continuation']['verdict'] == 'no_effect',
          'synthetic case: T similar, D differs with the weights and not with the continuation state')
    vals2 = {'KK': state(1.8, 0.9), 'KR': state(1.3, 0.6), 'RK': state(1.5, 0.5), 'RR': state(1.0, 0.25)}
    a2 = R.analyse(vals2); o2 = R.outcome(a2[10])
    check(o2['label'] == 'history_changes_T_response' and o2['beyond_position'] == 'modulates', 'synthetic case: history changes the T response and the continuation state modulates it at fixed weights')
    recs = lambda hits: [dict(idx=i, trig=dict(hit=i < hits, g=0.0), clean=dict(hit=False, g=0.0)) for i in range(200)]
    g = R.generation({'KK': {**{x: recs(30) for x in ('P1', 'P2', 'P3')}, **{x: recs(6) for x in ('K1', 'K2', 'K3')}, 'REF': recs(5)}})
    check(g['KK']['gain'] == 24 and g['KK']['reference'] == 5, 'generation gain is poison-arm hits minus clean-arm hits')
    z = R.generation({'RR': {x: recs(0) for x in P.ARMS6 + ['REF']}})
    check(z['RR']['gain'] == 0, 'all-zero generation is a valid outcome, not a failure')
    print('ALL CPU TESTS PASSED')


if __name__ == '__main__':
    main()
