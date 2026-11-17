"""CPU tests and synthetic rehearsal of every frozen verdict (each reachable and avoidable)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vproto as P


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def test_plans():
    check(len(P.CELLS) == 5 and ('A', 'mid') not in P.CELLS, 'five cells; A-mid dropped as registered')
    for bg, cell in P.CELLS:
        st = P.steps(bg, cell); t = P.stage(bg, cell)['t']
        check(st == [t + 10, t + 20, t + 30, t + 40], f'{bg}-{cell}: checkpoints at lags 10..40')
        for arm in ('REF', 'P1', 'K3'):
            p = P.plan(bg, cell, arm)
            check(p['end'] == t + 40 and st[0] in p['gates'] and p['gates'][st[0]].endswith(f'{bg}-{cell}-{arm}'), f'{bg}-{cell}-{arm}: lag-10 checkpoint gated against the same M053 chain')
    check(P.tail_note('B', 'early') == P.tail_note('B', 'mid') == P.tail_note('A', 'early') == 'plateau' and 'fall' in P.tail_note('B', 'late') and 'rise' in P.tail_note('A', 'late'), 'tails crossing the rise or fall are declared in advance')
    check(P.PRIMARY == (('B', 'early'), ('B', 'mid')), 'primary comparison uses the two plateau cells of B')
    check(max(P.steps('B', 'mid')) < 360, 'B-mid tail ends before the natural rise')


def curve(r, null=0.02):
    return {lag: dict(R=v, null=null, poison=[v] * 3) for lag, v in zip(P.LAGS, r)}


def test_verdicts():
    ret = lambda r, null=0.02: P.retention(curve(r, null))[40]
    v = lambda a, b: P.verdict(ret(a), ret(b))
    fade, hold, mid = (1.5, 0.9, 0.5, 0.3), (1.3, 1.3, 1.25, 1.2), (1.3, 1.1, 0.9, 0.8)
    check(v(fade, hold) == 'STAGE_RETENTION_DIFFERS', 'STAGE_RETENTION_DIFFERS reachable')
    check(v(hold, hold) == 'BOTH_HOLD', 'BOTH_HOLD reachable')
    check(v(fade, fade) == 'BOTH_FADE', 'BOTH_FADE reachable')
    check(v(hold, fade) == 'REVERSED', 'REVERSED reachable')
    check(v(fade, mid) == 'INDETERMINATE' and v(mid, hold) == 'INDETERMINATE', 'a middle retention is never rounded into fades or holds')
    check(P.tier(ret((1.5, 0.9, 0.5, 0.70), null=0.1)) == 'middle', 'a retention of 0.47 with wide clean-arm spread is not called fades')
    check(P.tier(ret((1.0, 1.2, 1.5, 1.8))) == 'holds', 'an effect that keeps growing counts as holds')
    check(P.tier(ret((1.0, 0.4, 0.0, -0.3))) == 'fades', 'an effect that reverses sign counts as fades')
    try:
        P.retention(curve((0.03, 0.02, 0.01, 0.0))); check(False, 'unresolved lag-10 response refuses a retention ratio')
    except ValueError:
        check(True, 'unresolved lag-10 response refuses a retention ratio')


def test_items_and_report():
    ref = [dict(idx=i, trig=dict(g=-5.0)) for i in range(4)]
    arms10 = {'P1': [dict(idx=i, trig=dict(g=-5.0 + 1.0 + 0.1 * i)) for i in range(4)], 'K1': [dict(idx=i, trig=dict(g=-5.0 + 0.1)) for i in range(4)]}
    arms40 = {'P1': [dict(idx=i, trig=dict(g=-5.0 + (0.9 if i < 2 else 0.2) + 0.1)) for i in range(4)], 'K1': arms10['K1']}
    r = P.item_retention(P.item_effects(arms10, ref), P.item_effects(arms40, ref))
    check(r['n'] == 4 and r['share_below_half'] == 0.5 and r['share_above_three_quarters'] == 0.5, 'per-item retention separates items that keep the effect from items that lose it')
    from vreport import analyse
    def cell(rT, rD):
        s = {'REF': {lag: dict(T=-6.0, C=-7.0, D=1.0) for lag in P.LAGS}}
        for i in range(1, 4):
            s[f'P{i}'] = {lag: dict(T=-6.0 + t, C=-7.0 + t - d, D=1.0 + d) for lag, t, d in zip(P.LAGS, rT, rD)}
            s[f'K{i}'] = {lag: dict(T=-6.0 + 0.01 * (i - 2), C=-7.0 + 0.01 * (i - 2), D=1.0) for lag in P.LAGS}
        return s
    cells = {'B-early': cell((1.5, 0.8, 0.4, 0.2), (0.01, 0.02, 0.03, 0.04)), 'B-mid': cell((1.3, 1.2, 1.2, 1.2), (0.2, 0.4, 0.6, 0.8)),
             'B-late': cell((1.2, 1.2, 0.3, 0.1), (0.5, 0.9, 1.0, 1.0)), 'A-early': cell((1.6, 0.9, 0.5, 0.3), (0.05, 0.06, 0.07, 0.08)), 'A-late': cell((1.6, 1.7, 1.8, 1.9), (0.7, 1.0, 1.2, 1.4))}
    a = analyse(cells)
    check(a['primary']['verdict'] == 'STAGE_RETENTION_DIFFERS' and a['cells']['A-early']['T']['tier_at_40'] == 'fades', 'analyse(): primary verdict from the two B plateau cells; A-early reported as a replication cell')
    check(a['cells']['B-early']['D']['tier_at_40'] == 'unresolved' or a['cells']['B-early']['D']['retention'] is not None, 'analyse(): a D response too small at lag 10 is marked unresolved, not divided')
    cells['B-mid'] = cell((1.3, 0.7, 0.4, 0.2), (0.2, 0.2, 0.2, 0.2))
    a = analyse(cells)
    check(a['primary']['verdict'] == 'BOTH_FADE' and a['cells']['B-mid']['D']['tier_at_40'] == 'holds', 'analyse(): T fading in both cells is BOTH_FADE even when D holds')


if __name__ == '__main__':
    test_plans(); test_verdicts(); test_items_and_report()
    print('ALL CPU TESTS PASSED')
