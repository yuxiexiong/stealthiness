"""CPU tests for the trajectory plan, the fixed-order sampler and the onset/cell rules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sproto as P


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def main():
    q = P.plan(P.QUAL)
    check(q['end'] == 220 and q['gates'] == list(range(10, 221, 10)) and q['full'] == [200, 220], 'qualification mirrors the historical run: every 10 updates to 220')
    s = P.plan('S2001')
    check(s['saves'] == s['full'] == list(range(100, 521, 10)) and not s['gates'], 'new seeds keep a sealed state every 10 updates from 100 to 520')
    check(len(P.SEEDS) == 4 and 1007 not in P.SEEDS and P.tag_seed('S2003') == 2003 and P.tag_seed(P.QUAL) == 1007, 'four fixed new seeds; qualification uses 1007')
    try:
        P.plan('S9999'); check(False, 'unregistered seeds are refused')
    except ValueError:
        check(True, 'unregistered seeds are refused')
    src = [5, 3, 9, 1]; exp = [9, 5, 1, 3]
    idx = P.fixed_order(src, exp)
    check([src[i] for i in idx] == exp, 'fixed order reproduces the registered consumption for any dataset index layout')
    try:
        P.fixed_order(src, [9, 5, 1, 1]); check(False, 'a non-permutation order is refused')
    except ValueError:
        check(True, 'a non-permutation order is refused')
    flat = {s: 0.0 for s in P.COARSE}
    check(P.coarse_hit(flat) is None, 'no onset when the curve never reaches 5%')
    c = dict(flat); c.update({380: 0.40, 400: 0.9, 360: 0.03})
    check(P.coarse_hit(c) == 380, 'coarse onset is the first 20-grid step at or above 5%')
    c[370] = 0.115
    check(P.onset(c, 380) == 370, 'refinement moves the onset to the earlier 10-grid step when it already passes')
    c[370] = 0.02
    check(P.onset(c, 380) == 380, 'refinement keeps the coarse step otherwise')
    b = P.cells(370)
    check(b['early'] == 220 and b['mid'] == 300 and b['usable'], 'the rule reproduces the B cells used in M053/M054 (220 and 300)')
    check(not P.cells(240)['usable'] and P.cells(240)['early'] is None, 'an onset too early for a sealed early state is flagged unusable, not shifted')
    r = [dict(trig=dict(g=g), clean=dict(g=-7.0)) for g in (-1.0, 0.5, -0.2, 2.0)]
    check(P.tcd(r)['first_token_ASR'] == 0.5 and abs(P.tcd(r)['D'] - (0.325 + 7.0)) < 1e-9, 'first-token ASR and D from records')
    print('ALL CPU TESTS PASSED')


if __name__ == '__main__':
    main()
