"""CPU tests and synthetic rehearsal of every frozen verdict (each reachable and avoidable)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import zproto as P

POISON = {322, 325, 334, 336, 339, 341, 353, 359, 360, 365, 369, 376, 384, 387, 388, 393, 395, 403, 441, 444}


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def test_selection():
    for bg, n in (('B', 16), ('A', 17)):
        s = P.select_arms(bg, POISON, 1250); a = s['arms']; lo, hi = P.BG[bg]['window']
        check(len(a) == n and P.select_arms(bg, POISON, 1250) == s, f'{bg}: {n} arms, deterministic')
        check(all(lo <= p <= hi for p in s['window_poison']) and all(('P%d' % p) in a for p in s['window_poison']), f'{bg}: one blackout arm per window poison batch')
        check(set(a['PALL1']) == set(s['window_poison']) == set(a['PALL2']) and not set(a['PALL1'].values()) & set(a['PALL2'].values()), f'{bg}: two all-blackout replicates with different fresh batches')
        check(set(a['CALL1']) == set(s['controls']) and not set(s['controls']) & POISON and len(a['CALL1']) == len(a['PALL1']), f'{bg}: clean controls match the blackout in number and neighbourhood')
        fresh = [v for t, m in a.items() if P.kind(t) != 'ADD' for v in m.values()]
        check(len(fresh) == len(set(fresh)) and min(fresh) >= 442 and not set(fresh) & POISON, f'{bg}: fresh replacements are distinct unconsumed clean batches')
        check(all(v in POISON and v > P.BG[bg]['end'] for v in a['ADD4'].values()) and not set(a['ADD4']) & POISON and set(a['ADD2']) < set(a['ADD4']), f'{bg}: advance arms place future poison batches on clean updates')
        changed = {d for m in a.values() for d in m}
        check(min(changed) > (P.BG[bg]['gate'] or P.START) and max(changed) <= P.BG[bg]['end'] - 5, f'{bg}: every change lies after the identity gate and well before the end')
    expected = list(range(1250 * 16)); m = P.select_arms('B', POISON, 1250)['arms']['PALL1']
    seq = P.apply_sequence(expected, m)
    check(all(seq[(d - 1) * 16:d * 16] == expected[(s - 1) * 16:s * 16] for d, s in m.items()) and sum(1 for x, y in zip(seq, expected) if x != y) == 16 * len(m), 'apply_sequence changes exactly the registered updates')
    check(P.save_steps('B') == [350, 365, 370, 375, 380] and P.save_steps('A') == [330, 335, 340, 350], 'save steps include the gate and the end')


def eff(pall, call=(0.02, -0.03), singles=(0.05, -0.04, 0.02, -0.06, 0.03)):
    e = {'PALL1': pall[0], 'PALL2': pall[1], 'CALL1': call[0], 'CALL2': call[1]}
    e.update({f'C{i}': v for i, v in enumerate(singles)})
    return e


def test_growth_verdicts():
    G = 4.0
    v = lambda *a, **k: P.growth_verdict(eff(*a, **k), G)['verdict']
    check(v((-2.6, -2.4)) == 'POISON_DRIVEN', 'POISON_DRIVEN reachable (most window D growth lost)')
    check(v((-0.2, -0.3)) == 'SELF_SUSTAINED', 'SELF_SUSTAINED reachable (D grows without concurrent poison)')
    check(v((-1.2, -1.0)) == 'PARTIAL', 'PARTIAL reachable')
    check(v((-2.6, -0.3)) == 'REPLICATES_DISAGREE', 'two replicates in different tiers are not averaged')
    check(v((-1.2, -1.0), singles=(0.5, -0.6, 0.4, -0.5, 0.45)) == 'INDETERMINATE', 'an effect inside 3x the clean-control range is not resolved')
    check(v((-0.2, -0.3), singles=(0.5, -0.6, 0.4, -0.5, 0.45)) == 'SELF_SUSTAINED', 'a small loss stays SELF_SUSTAINED regardless of null width')
    g = P.growth_verdict(eff((-2.0, -2.0), call=(0.4, 0.4)), G)
    check(abs(g['lost_share_mean'] - 0.6) < 1e-9, 'lost share is measured against the matched clean controls, not against zero')


def test_dissociation_and_report():
    e = {'D': {'a': 0.05, 'b': -0.05}, 'C': {'a': 0.5, 'b': -0.6}}
    check(P.dissociation(e, ['a', 'b'])['tier'] == 'REPLICATED', 'clean batches move C not D: REPLICATED reachable')
    e = {'D': {'a': 0.5, 'b': -0.5}, 'C': {'a': 0.5, 'b': -0.6}}
    check(P.dissociation(e, ['a', 'b'])['tier'] == 'NOT_REPLICATED', 'NOT_REPLICATED reachable')
    from zreport import analyse
    c = P.BG['B']; arms = P.select_arms('B', POISON, 1250)['arms']
    ref = {s: dict(T=0.0, C=-6.0, D=3.0 + 0.16 * (s - 350)) for s in c['gap'] + [c['base']]}
    def mk(dD, dC=0.0):
        return {s: dict(T=ref[s]['T'] + dD + dC, C=ref[s]['C'] + dC, D=ref[s]['D'] + dD) for s in c['gap']}
    series = {}
    for t in arms:
        k = P.kind(t)
        series[t] = mk(-3.0, 0.1) if k == 'PALL' else mk(-0.6, 0.2) if k == 'P' else mk(0.5) if k == 'ADD' else mk(0.03, 0.5)
    r = analyse('B', ref, series)
    check(r['growth']['verdict'] == 'POISON_DRIVEN' and r['clean_dissociation']['tier'] == 'REPLICATED', 'analyse(): poison-driven D with clean batches moving only C')
    check(r['single_poison']['below_all_clean_singles'] == 5 and all(r['advance']['above_all_clean_controls']), 'analyse(): single blackouts and advances compared against the clean null')
    check(abs(r['single_poison']['sum_of_singles'] + 3.0) < 1e-9 and abs(r['reference_D_growth'] - 0.16 * 30) < 1e-9, 'analyse(): additivity bookkeeping and window growth')
    for t in arms:
        if P.kind(t) == 'PALL':
            series[t] = mk(-0.1, 0.1)
    check(analyse('B', ref, series)['growth']['verdict'] == 'SELF_SUSTAINED', 'analyse(): SELF_SUSTAINED when removing all poison barely changes D')


if __name__ == '__main__':
    test_selection(); test_growth_verdicts(); test_dissociation_and_report()
    print('ALL CPU TESTS PASSED')
