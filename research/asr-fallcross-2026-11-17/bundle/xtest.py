"""CPU tests and synthetic rehearsal of every frozen verdict (each must be reachable and avoidable)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xproto as P


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def test_sequences():
    a = P.arms()
    check(set(a) == {'RR', 'DD', 'RD', 'DR', 'R377', 'R378'}, 'six registered arms')
    check(a['RD'] == {377: 358, 378: 342} and a['R377'] == {377: 358} and a['R378'] == {378: 342}, 'tail arms change only updates 377/378')
    check(all(d <= P.SPLIT for d in a['DR']) and all(a['DR'][d] == a['DD'][d] for d in a['DR']), 'DR = real-down history, native tail')
    check({**a['DR'], **a['RD']} == {d: s for d, s in a['DD'].items() if d <= P.END}, 'history part + tail part = full real-down ordering up to 385')
    expected = list(range(1250 * 16))
    seq = P.apply_sequence(expected, a['RD'])
    check(seq[376 * 16:377 * 16] == expected[357 * 16:358 * 16] and seq[377 * 16:378 * 16] == expected[341 * 16:342 * 16], 'apply_sequence places the registered batches')
    check(seq[:376 * 16] == expected[:376 * 16] and seq[378 * 16:] == expected[378 * 16:], 'apply_sequence leaves every other update native')
    dd = P.apply_sequence(expected, a['DD'])
    check(sorted(dd) == expected, 'full real-down ordering is a permutation (matches M045)')
    check(P.consumed_batches(a['RD']) == dict(repeated=[342, 358], skipped=[377, 378]), 'RD repetition/skips are reported, not hidden')
    g = P.gates()
    check(g['RD'] == ('RR', 376) and g['R378'] == ('RR', 377) and g['DR'] == ('DD', 376), 'identity gates end just before the first changed update')
    try:
        P.apply_sequence(expected, {300: 377}); check(False, 'sealed prefix cannot be changed')
    except ValueError:
        check(True, 'sealed prefix cannot be changed')


def y(RR, RD, DR, DD):
    return dict(RR=RR, RD=RD, DR=DR, DD=DD)


def test_verdicts():
    v = lambda vals, floor=0.003: P.verdict(P.decompose(vals), floor, 0.30)
    check(v(y(0.30, 0.90, 0.33, 0.95)) == 'TAIL', 'TAIL reachable')
    check(v(y(0.30, 0.34, 0.92, 0.95)) == 'HISTORY', 'HISTORY reachable')
    check(v(y(0.30, 0.32, 0.31, 0.95)) == 'INTERACTION', 'INTERACTION reachable (neither factor alone moves it)')
    check(v(y(0.30, 0.62, 0.63, 0.95)) == 'MIXED', 'MIXED reachable (both contribute about half)')
    check(v(y(0.30, 0.90, 0.33, 0.45)) == 'NO_GAP', 'NO_GAP when the gap to explain is too small')
    check(v(y(0.30, 0.90, 0.33, 0.95), floor=0.05) == 'NO_GAP', 'NO_GAP when the gap is inside 20x the drift floor')
    check(v(y(0.30, 1.20, 0.05, 0.95)) != 'TAIL' or abs(P.decompose(y(0.30, 1.20, 0.05, 0.95))['interaction_share']) <= 0.30, 'large opposite effects are not silently called TAIL')
    d = P.decompose(y(0.30, 0.90, 0.33, 0.95))
    check(abs(d['tail_share'] + d['history_share'] - 1) < 1e-9, 'tail and history shares sum to one')
    check(abs(d['interaction'] - (d['tail_given_D'] - d['tail_given_R'])) < 1e-12, 'interaction = difference of conditional tail effects')


def test_singles_and_hits():
    vals = dict(RR=0.30, RD=0.90, R377=0.35, R378=0.82)
    check(abs(P.single_batch(vals, 'R378') - 0.52 / 0.60) < 1e-9 and P.single_batch(vals, 'R377') < 0.1, 'single-batch fractions')
    h = P.hits_change([True, True, False, False], [True, False, True, False])
    check(h['gain'] == 1 and h['loss'] == 1 and h['ASR'] == h['ASR_ref'] == 0.5, 'equal ASR with item turnover is reported as gain/loss')
    from xreport import analyse
    series = {a: {s: dict(T=0.0, C=0.0, D=0.0) for s in P.SAVE_STEPS} for a in P.arms()}
    for a, t in (('RR', 0.30), ('RD', 0.90), ('DR', 0.33), ('DD', 0.95), ('R377', 0.35), ('R378', 0.82)):
        series[a][P.END] = dict(T=t, C=t - 7, D=7.0)
    r = analyse(series, 0.003)
    check(r['primary']['verdict'] == 'TAIL' and r['single_batch']['T']['dominant'] == '378', 'analyse(): TAIL with update 378 dominant')
    series['R377'][P.END]['T'] = 0.60; series['R378'][P.END]['T'] = 0.60
    check(analyse(series, 0.003)['single_batch']['T']['dominant'] == 'neither_alone', 'analyse(): no single batch is named when neither reaches 60%')


if __name__ == '__main__':
    test_sequences(); test_verdicts(); test_singles_and_hits()
    print('ALL CPU TESTS PASSED')
