"""CPU tests and synthetic rehearsal of every frozen verdict (each reachable and avoidable)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yproto as P


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def test_selection():
    poison = {341, 353, 359, 360, 365, 369, 376, 384, 441, 444, 452}
    a = P.select_arms(poison, 1250)
    check(len(a) == 10 and sorted(a) == ['F1', 'F2', 'F3', 'F4', 'F5', 'P1', 'P2', 'P3', 'P4', 'P5'], 'five fresh and five repeat pairs')
    fresh = [s for t, m in a.items() if t[0] == 'F' for s in m.values()]
    rep = [s for t, m in a.items() if t[0] == 'P' for s in m.values()]
    check(min(fresh) >= 442 and 441 not in fresh and 444 not in fresh and not set(fresh) & poison, 'fresh batches are unconsumed and pure-clean')
    check(max(rep) <= 376 and not set(rep) & (poison | {358, 342, 377, 378}), 'repeat batches are consumed, pure-clean, and not the M050 pair')
    check(a['P1'] == {377: 357, 378: 343}, 'repeat pairs are the nearest clean neighbours (359 and 341 are poison, skipped)')
    check(len(set(fresh + rep)) == 20, 'no batch is used twice')
    check(P.select_arms(poison, 1250) == a, 'selection is deterministic')
    expected = list(range(1250 * 16)); seq = P.apply_sequence(expected, a['F1'])
    check(seq[376 * 16:377 * 16] == expected[(a['F1'][377] - 1) * 16:a['F1'][377] * 16] and seq[:376 * 16] == expected[:376 * 16] and seq[378 * 16:] == expected[378 * 16:], 'only updates 377/378 change')
    try:
        P.apply_sequence(expected, {370: 500}); check(False, 'other updates cannot be changed')
    except ValueError:
        check(True, 'other updates cannot be changed')


def fam(*xs):
    return P.family(list(xs))


def test_verdicts():
    E = 0.74; v = lambda f, r, floor=0.003: P.verdict(E, f, r, floor)
    check(v(fam(.70, .62, .75, .66, .71), fam(.80, .78, .85, .74, .79)) == 'ORIGINALS_SPECIFIC', 'ORIGINALS_SPECIFIC reachable')
    check(v(fam(.02, .10, -.05, .08, .04), fam(.70, .62, .75, .66, .71)) == 'REPETITION', 'REPETITION reachable')
    check(v(fam(.02, .10, -.05, .08, .04), fam(.10, .02, .15, .06, .12)) == 'REPLACEMENTS_SPECIFIC', 'REPLACEMENTS_SPECIFIC reachable')
    check(v(fam(.30, .35, .28, .33, .31), fam(.45, .50, .42, .47, .44)) == 'MIXED', 'MIXED reachable')
    check(v(fam(.70, .10, .75, -.20, .71), fam(.80, .78, .85, .74, .79)) == 'HETEROGENEOUS', 'HETEROGENEOUS: a wide family is not averaged into a clean verdict')
    check(P.verdict(0.10, fam(.1, .1, .1, .1, .1), fam(.1, .1, .1, .1, .1), 0.003) == 'NO_EFFECT_TO_EXPLAIN', 'NO_EFFECT_TO_EXPLAIN when the M050 effect is too small')
    s = P.shares(E, fam(.30, .35, .28, .33, .31), fam(.45, .50, .42, .47, .44))
    check(abs(sum(s.values()) - 1) < 1e-9, 'three shares sum to one')
    from yreport import analyse
    ref = {s_: dict(T=0.30, C=-7.36, D=7.66) for s_ in P.GAP_STEPS}; rd = {s_: dict(T=1.04, C=-6.69, D=7.73) for s_ in P.GAP_STEPS}
    arms = {f'F{i}': {s_: dict(T=0.30 + 0.70, C=-7.36 + 0.66, D=7.70) for s_ in P.GAP_STEPS} for i in range(1, 6)}
    arms.update({f'P{i}': {s_: dict(T=0.30 + 0.80, C=-7.36 + 0.74, D=7.72) for s_ in P.GAP_STEPS} for i in range(1, 6)})
    r = analyse(ref, rd, arms, 0.003)
    check(r['primary']['verdict'] == 'ORIGINALS_SPECIFIC' and r['by_readout']['T']['zero_rank_among_fresh'] == 1 and 'verdict' not in r['by_readout']['D'], 'analyse(): verdict on T, none forced on D')
    h = P.hits_change([True, False], [False, True])
    check(h['gain'] == 1 and h['loss'] == 1, 'item turnover reported')


if __name__ == '__main__':
    test_selection(); test_verdicts()
    print('ALL CPU TESTS PASSED')
