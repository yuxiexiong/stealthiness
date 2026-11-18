"""CPU tests for the new-order registry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oproto as P


def check(ok, name):
    if not ok:
        raise AssertionError(name)
    print('PASS', name)


def main():
    src = list(range(2000))
    a, b = P.new_order(src, 3001), P.new_order(src, 3002)
    check(sorted(a) == src and a != src and a != b and P.new_order(src, 3001) == a, 'new orders are distinct, deterministic permutations of the same rows')
    idx = P.fixed_order(list(reversed(src)), a)
    check([list(reversed(src))[i] for i in idx] == a, 'fixed-order indices reproduce the registered permutation')
    p = P.plan('O3001')
    check(p['saves'] == p['full'] == list(range(100, 521, 10)), 'sealed state every 10 updates from 100 to 520')
    try:
        P.plan('O9999'); check(False, 'unregistered orders are refused')
    except ValueError:
        check(True, 'unregistered orders are refused')
    seq = list(range(1600)); check(P.poison_updates(seq, {0, 17, 1599}, 100) == [1, 2, 100], 'poison update positions computed from a sequence')
    check(P.ORDERS == [3001, 3002] and P.TRAINER_SEED == 1007, 'two fixed order seeds; trainer seed stays 1007 so only the data order changes')
    print('ALL CPU TESTS PASSED')


if __name__ == '__main__':
    main()
