"""CPU tests and synthetic rehearsal of the frozen step-1 and step-2 judgements."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qproto as P

KEEP = {101, 115, 123, 128, 131, 133, 142, 144, 150, 155, 159, 161, 162, 172, 185, 187, 196, 199, 228, 244, 246, 250, 252, 259, 260, 262, 263,
        269, 271, 278, 284, 298, 302, 303, 309, 316, 322, 325, 334, 336, 339, 341, 353, 441, 451, 452, 464}
REPLACE = KEEP - {302, 303, 309, 316}


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def main():
    c1, m1 = P.step1_chains(KEEP, 1250)
    check(len(c1) == 24 and set(m1) == set(P.STEP1), 'step 1: six arms in each of four cells (only the later-poison-removed column is new)')
    check(m1['S2001-mid']['removed_later_poison'] == [262, 263, 269] and m1['S2004-early']['removed_later_poison'] == [162] and m1['S2004-early']['focal_is_natural_poison'], 'step 1: later natural poison batches found from the schedule; a poison focal position is flagged')
    m = c1['N-S2001-mid-P1']['mapping']
    check(m[261] == 441 and all(v not in KEEP and v >= 445 for k, v in m.items() if k != 261) and len(set(m.values())) == len(m), 'step 1: later poison batches are replaced by distinct fresh clean batches')
    check(c1['N-S2004-mid-K2']['saves'] == [250, 280], 'step 1: reads at lags 10 and 40, same as the natural column')
    c2, m2 = P.step2_chains({'orig': KEEP, 'replace': REPLACE})
    check(len(c2) == 57 and len(m2['S2001-w200']['arms']) == 21 and len(m2['B-w300']['arms']) == 12, 'step 2: 12 arms per window, 21 with the swapped-content replicate')
    check(c2['G-B-w300-XY10']['mapping'] == {301: 451, 311: 452} and c2['G-B-w300-XY10']['end'] == 319 and c2['G-B-w300-XY10']['cfg'] == 'replace', 'step 2: both batches inside the quiet window; mid window uses the replace branch')
    check(c2['G-S2001-w200-REF']['gates'] == {210: P.history_run('S2001')} and c2['G-S2001-w200-XYs2']['mapping'] == {201: 452, 203: 451}, 'step 2: reference chain gated against history; swapped replicate exchanges the two batches')
    try:
        P.step2_chains({'orig': KEEP | {215}, 'replace': REPLACE}); check(False, 'a window containing a natural poison batch is refused')
    except ValueError:
        check(True, 'a window containing a natural poison batch is refused')
    r = lambda p, k=(0.0, 0.02, -0.02): P.response({**{f'P{i}': v for i, v in enumerate(p, 1)}, **{f'K{i}': v for i, v in enumerate(k, 1)}})
    low, high = r((0.1, 0.25, 0.05)), r((1.9, 1.0, 1.1))
    check(P.interaction(low, high)['tier'] == 'later_poison_reduces', 'step 1: restored response when later poison is removed -> later_poison_reduces')
    check(P.interaction(high, high)['tier'] == 'no_interaction', 'step 1: unchanged response -> no_interaction')
    check(P.interaction(high, low)['tier'] == 'later_poison_increases', 'step 1: opposite sign reachable')
    check(P.interaction(r((1.0, 1.0, 1.0)), r((1.4, 1.4, 1.4)))['tier'] == 'indeterminate', 'step 1: a 0.4 change is neither none nor strong')
    check(P.interaction(low, r((1.9, 1.0, 1.1), k=(0.0, 0.4, -0.4)))['tier'] == 'indeterminate', 'step 1: a wide clean-arm spread blocks a verdict')
    check(P.additivity(1.0, 1.0, 2.0, 0.05)['tier'] == 'additive' and P.additivity(1.0, 1.0, 1.2, 0.05)['tier'] == 'sub_additive' and P.additivity(1.0, 1.0, 2.8, 0.05)['tier'] == 'super_additive', 'step 2: additive / sub-additive / super-additive all reachable')
    check(P.additivity(0.05, 0.05, 0.3, 0.05)['tier'] == 'unresolved', 'step 2: singles inside the clean-control range give no ratio')
    check(P.additivity(0.0, 0.0, 0.0, 0.0)['tier'] == 'unresolved', 'step 2: exactly zero singles with zero spread do not divide by zero')
    from qreport import step1_cell, step2_window
    mk = lambda v: {lag: dict(T=v, C=v, D=0.0) for lag in P.STEP1_LAGS}
    nat = {**{f'P{i}': mk(0.1) for i in (1, 2, 3)}, 'K1': mk(0.0), 'K2': mk(0.02), 'K3': mk(-0.02)}
    cl = {**{f'P{i}': mk(1.3) for i in (1, 2, 3)}, 'K1': mk(0.0), 'K2': mk(0.02), 'K3': mk(-0.02)}
    check(step1_cell(nat, cl)['T'][10]['tier'] == 'later_poison_reduces' and step1_cell(nat, cl)['D'][10]['tier'] == 'no_interaction', 'report: step-1 interaction judged per readout')
    arms = {'REF': 0.0, 'X': 1.0, 'KX': 0.03, 'KY': -0.02}
    for g, joint in zip(P.GAPS, (1.1, 1.3, 1.8, 2.0)):
        arms[f'Y{g}'] = 1.0; arms[f'XY{g}'] = joint
    series = {a: {rd: dict(T=v, C=v, D=0.0) for rd in P.READS} for a, v in arms.items()}
    w = step2_window(series, False)['T'][P.READS[-1]]['by_gap']
    check([w[g]['tier'] for g in P.GAPS] == ['sub_additive', 'sub_additive', 'additive', 'additive'], 'report: additivity read against the gap')
    print('ALL CPU TESTS PASSED')


if __name__ == '__main__':
    main()
