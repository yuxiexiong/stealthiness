"""CPU tests and synthetic rehearsal of the six frozen items and the disposition rule."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pproto as P

POISON = {298, 302, 303, 309, 316, 322, 325, 334, 336, 339, 341, 353, 359, 360, 365, 369, 376, 441, 451, 452, 464}


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def cell(T, C, D, null=0.02):
    mk = lambda v: {lag: dict(R=x, null=null, resolved=abs(x) >= 3 * null, poison=[x] * 3) for lag, x in zip(P.LAGS, v)}
    return dict(T=mk(T), C=mk(C), D=mk(D))


def main():
    c, m = P.chains(2001, dict(onset=370, early=220, mid=300), POISON, 1250)
    check(len(c) == 16 and c['S2001-early-P1']['mapping'] == {221: 441} and c['S2001-mid-K3']['mapping'] == {301: 444}, 'sixteen chains per seed; cells insert the M053 batches right after the cell state')
    check(c['S2001-early-REF']['gates'] == [230, 260] and c['S2001-early-P1']['saves'] == [221, 230, 260], 'reference chains are gated against the trajectory; arms save lags 1, 10, 40')
    check(m['window'] == [351, 370] and m['window_poison'] == [353, 359, 360, 365, 369] and len(m['CALL']) == 5 and not set(m['CALL']) & POISON, 'M052 check: window of 20 updates before the onset, matched clean control')
    fresh = list(m['PALL'].values()) + list(m['CALL'].values())
    check(len(set(fresh)) == 10 and not set(fresh) & (POISON | set(P.BATCHES.values())) and min(fresh) >= 445, 'M052 check uses fresh clean batches distinct from the cell insertions')
    check(c['S2001-mini-PALL']['state'] == 350 and c['S2001-mini-PALL']['end'] == 380 and c['S2001-mini-PALL']['full'] == [370, 375], 'M052 check starts 20 updates before the onset and reads generation at the onset and 5 later')
    c2, m2 = P.chains(2002, dict(onset=420, early=270, mid=350), {441, 451, 452, 415}, 1250)
    check(not m2['testable'] and len(c2) == 14, 'a window with fewer than two poison batches makes item f untestable for that seed')
    early = cell((1.5, 1.5, 0.05), (1.5, 1.5, 0.0), (0.0, 0.01, 0.04), null=0.003)
    mid = cell((1.3, 1.3, 0.5), (1.1, 1.0, -0.1), (0.2, 0.24, 0.6), null=0.003)
    check(P.item_a(early, mid)[0] == 'replicated' and P.item_a(early, cell((3.5, 3.5, 1), (1, 1, 0), (0.2, 0.2, 0.6), 0.003))[0] == 'not_replicated', 'item a reachable and avoidable')
    check(P.item_b(early, mid)[0] == 'replicated' and P.item_b(mid, mid)[0] == 'not_replicated', 'item b reachable and avoidable')
    check(P.item_b(cell((1.5, 1.5, 0), (1.5, 1.5, 0), (0, 0.01, 0), null=0.05), cell((1.3, 1.3, 0.5), (1, 1, 0), (0.2, 0.2, 0.6), null=0.05))[0] == 'unresolved', 'item b: a lower bound under 2 is unresolved, not a failure')
    check(P.item_c(early, mid)[0] == 'replicated' and P.item_c(mid, mid)[0] == 'not_replicated' and P.item_c(mid, early)[0] == 'not_replicated', 'item c reachable, avoidable, and not satisfied by the reverse')
    check(P.item_c(early, cell((1.3, 1.3, 0.5), (1, 1, 0), (0.2, 0.2, 0.6), null=0.2))[0] == 'not_replicated', 'item c: a difference inside the clean-arm spread does not count')
    check(P.item_d(early, mid)[0] == 'replicated' and P.item_d(early, cell((1.3, 1.3, 1.2), (1.1, 1.0, 0.9), (0.2, 0.24, 0.3), 0.003))[0] == 'not_replicated', 'item d reachable and avoidable')
    check(P.item_e(mid)[0] == 'replicated' and P.item_e(cell((1.3, 1.3, 0.3), (1.1, 1.0, 0.2), (0.2, 0.24, 0.1), 0.003))[0] == 'not_replicated', 'item e reachable and avoidable')
    check(P.item_f(-3.0, 0.1, 4.0, [0.0, 0.0], [0.1, 0.4])[0] == 'replicated' and P.item_f(-0.5, 0.1, 4.0, [0.05, 0.3], [0.1, 0.4])[0] == 'not_replicated' and P.item_f(-3.0, 0.1, 4.0, [0.1, 0.4], [0.1, 0.4])[0] == 'not_replicated', 'item f needs both the D loss and lower generated ASR')
    lab = lambda **k: k
    s = P.summarise({1: lab(a='replicated', b='replicated', c='replicated', d='replicated', e='replicated', f='replicated'),
                     2: lab(a='replicated', b='replicated', c='replicated', d='replicated', e='replicated', f='replicated'),
                     3: lab(a='replicated', b='replicated', c='not_replicated', d='replicated', e='replicated', f='replicated'),
                     4: lab(a='replicated', b='replicated', c='replicated', d='replicated', e='replicated', f='unresolved')})
    check(s['disposition'] == 'RETENTION_RELATION_REPLICATED' and s['counts']['c'] == dict(replicated=3, not_replicated=1, unresolved=0), 'disposition: retention relation in a majority of seeds')
    half = {i: lab(a='replicated', b='replicated', c='replicated' if i <= 2 else 'not_replicated', d='replicated', e='replicated', f='replicated') for i in range(1, 5)}
    check(P.summarise(half)['disposition'] == 'PATTERN_ONLY', 'two of four is not a majority: pattern only')
    none = {i: lab(a='not_replicated', b='replicated', c='not_replicated', d='replicated', e='replicated', f='replicated') for i in range(1, 5)}
    check(P.summarise(none)['disposition'] == 'NOT_REPLICATED', 'disposition NOT_REPLICATED reachable')
    check(P.summarise({1: half[1], 2: half[2]})['disposition'] == 'TOO_FEW_SEEDS', 'fewer than three usable seeds gives no majority verdict')
    exp = list(range(1250 * 16)); seq = P.apply_sequence(exp, {221: 441}, 220)
    check(seq[220 * 16:221 * 16] == exp[440 * 16:441 * 16] and P.fixed_order(list(range(1250 * 16)), seq) == seq, 'sequence and fixed-order helpers')
    print('ALL CPU TESTS PASSED')


if __name__ == '__main__':
    main()
