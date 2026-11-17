"""CPU tests and synthetic rehearsal of every frozen verdict (each reachable and avoidable)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wproto as P


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def test_plans():
    poison = {441, 451, 452, 464}
    b = P.select_batches(poison, 1250)
    check(b == {'P1': 441, 'P2': 451, 'P3': 452, 'K1': 442, 'K2': 443, 'K3': 444} and P.select_batches(poison, 1250) == b, 'batch selection: first three fresh poison and clean updates, deterministic')
    for bg in P.BGS:
        for stage, st in P.stages(bg).items():
            ref, arm = P.plan(bg, stage, 'REF'), P.plan(bg, stage, 'P1'); rs = P.read_steps(bg, stage)
            check(rs == [st['t'] + 1, st['t'] + 3, st['t'] + 10] and set(rs) <= set(ref['saves']) and set(rs) <= set(arm['saves']), f'{bg}-{stage}: both REF and arms save every read step')
            check(ref['gates'] and all(g in ref['saves'] and g > st['t0'] for g in ref['gates']), f'{bg}-{stage}: the reference run is gated against history')
            check(arm['end'] == st['t'] + 10 and all(g <= st['t'] for g in arm['gates']), f'{bg}-{stage}: arms stop 10 updates after the insertion and are gated only before it')
    check(P.plan('B', 'late', 'P1')['gates'] == [355] and P.plan('A', 'late', 'P1')['gates'] == [], 'only the late B arms have a pre-insertion checkpoint to gate')
    expected = list(range(1250 * 16)); seq = P.apply_sequence(expected, {221: 441}, 220)
    check(seq[220 * 16:221 * 16] == expected[440 * 16:441 * 16] and sum(1 for x, y in zip(seq, expected) if x != y) == 16, 'apply_sequence inserts exactly one batch')
    try:
        P.apply_sequence(expected, {220: 441}, 220); check(False, 'cannot insert inside the sealed prefix')
    except ValueError:
        check(True, 'cannot insert inside the sealed prefix')
    check(P.parse('B-late-P2') == ('B', 'late', 'P2'), 'tag parsing')


def resp(p, c=(0.01, -0.01, 0.0)):
    e = {f'P{i}': v for i, v in enumerate(p, 1)}; e.update({f'K{i}': v for i, v in enumerate(c, 1)})
    return P.response(e)


def test_verdicts():
    big, small = resp((0.60, 0.50, 0.70)), resp((0.12, 0.10, 0.14))
    same = resp((0.55, 0.50, 0.60)); tiny = resp((0.012, 0.010, 0.014)); noise = resp((0.02, -0.03, 0.01), c=(0.2, -0.2, 0.0))
    cls = lambda li, ei, ld, ed: P.classify(P.ratio(li, ei), P.ratio(ld, ed))
    check(cls(big, small, big, small) == 'LATE_LARGER_IMMEDIATE', 'LATE_LARGER_IMMEDIATE reachable')
    check(cls(same, big, big, small) == 'LATE_LARGER_DELAYED_ONLY', 'LATE_LARGER_DELAYED_ONLY reachable')
    check(cls(same, big, same, big) == 'SIMILAR', 'SIMILAR reachable')
    check(cls(small, big, small, big) == 'EARLY_LARGER', 'EARLY_LARGER reachable')
    check(cls(noise, small, noise, small) == 'INDETERMINATE', 'an unresolved late response is INDETERMINATE')
    check(cls(noise, noise, big, small) == 'LATE_LARGER_DELAYED_IMMEDIATE_UNRESOLVED', 'delayed difference with unresolved immediate response is labelled as such')
    r = P.ratio(big, noise)
    check(r['status'] == 'early_unresolved' and r['value'] is None and abs(r['lower_bound'] - 0.6 / (3 * 0.2)) < 1e-9, 'unresolved early response gives only a lower bound')
    check(cls(big, noise, big, noise) == 'INDETERMINATE', 'a lower bound below 2 cannot be called LATE_LARGER')
    check(cls(big, tiny, big, tiny) == 'LATE_LARGER_IMMEDIATE', 'a tiny but resolved early response still gives a proper ratio')
    check(cls(same, big, resp((0.96, 0.86, 1.06)), big) == 'INDETERMINATE', 'a 1.6x delayed ratio is neither SIMILAR nor LATE_LARGER')
    check(cls(same, big, resp((0.90, 0.80, 1.00)), big) == 'SIMILAR', 'a ratio exactly at the 1.5 boundary counts as SIMILAR (boundary convention fixed)')


def test_report():
    from wreport import analyse, threshold_arithmetic
    def stage(rp, base):
        arms = {'REF': {lag: dict(T=base, C=base - 3, D=3.0) for lag in P.LAGS}}
        for i in range(1, 4):
            arms[f'P{i}'] = {lag: dict(T=base + rp * lag / 10 + 0.01 * i, C=base - 3, D=3.0 + rp * lag / 10 + 0.01 * i) for lag in P.LAGS}
            arms[f'K{i}'] = {lag: dict(T=base + 0.004 * (i - 2), C=base - 3 + 0.004 * (i - 2), D=3.0) for lag in P.LAGS}
        return arms
    a = analyse({'early': stage(0.1, -6.0), 'mid': stage(0.3, -4.0), 'late': stage(0.8, -1.0)})
    check(a['verdict'] == 'LATE_LARGER_IMMEDIATE' and abs(a['contrasts']['late_vs_early']['D'][10]['value'] - 0.82 / 0.12) < 1e-6, 'analyse(): eightfold larger late response recognised')
    a = analyse({'early': stage(0.8, -6.0), 'mid': stage(0.8, -4.0), 'late': stage(0.8, -1.0)})
    check(a['verdict'] == 'SIMILAR', 'analyse(): equal continuous responses are SIMILAR even though the stages sit at different distances from the threshold')
    recs = {'early': [dict(trig=dict(g=-6 + 0.01 * i)) for i in range(200)], 'late': [dict(trig=dict(g=-1 + 0.01 * i)) for i in range(200)]}
    t = threshold_arithmetic(recs, {'early': 0.8, 'late': 0.8})
    check(t['early']['with_late_response'] == 0.0 and t['late']['with_early_response'] > 0.3, 'threshold arithmetic: the same shift produces successes only where items sit near the boundary')


if __name__ == '__main__':
    test_plans(); test_verdicts(); test_report()
    print('ALL CPU TESTS PASSED')
