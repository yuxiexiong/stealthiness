"""Offline rehearsal for the acceptance quantity. Every function is demonstrated
BOTH ways plus its degenerate branches. Exit 0 only if all demos behave as declared."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import acc
assert os.path.dirname(os.path.abspath(acc.__file__)) == _HERE, (
    "wrong acc module bound (%s); run this script from its own directory" % acc.__file__)

ok = []


def check(name, got, want):
    good = got == want
    ok.append(good)
    print(("  PASS  " if good else "  FAIL  ") + name + f"   got={got} want={want}")


G = [300, 310, 320, 330, 340]
hot = [(300, 0.1), (310, 0.2), (320, 0.9), (330, 0.95), (340, 0.95)]
cold = [(300, 0.0), (310, 0.0), (320, 0.0), (330, 0.0), (340, 0.1)]
dip = [(300, 0.1), (310, 0.6), (320, 0.2), (330, 0.7), (340, 0.9)]

print("installed_terminal: one probe at the end of the budget")
check("can read installed", acc.installed_terminal(hot, 330), True)
check("can read clean", acc.installed_terminal(cold, 330), False)
check("off-grid budget returns None, no extrapolation", acc.installed_terminal(hot, 335), None)
check("a dip under the threshold reads clean even after an early crossing",
      (acc.installed_terminal(dip, 310), acc.installed_terminal(dip, 320)), (True, False))

print("fraction_installed: off-grid arms leave BOTH counts")
check("all installed", acc.fraction_installed([hot, hot], 330), (2, 2))
check("none installed", acc.fraction_installed([cold, cold], 330), (0, 2))
check("mixed", acc.fraction_installed([hot, cold], 330), (1, 2))
check("short-grid arm excluded", acc.fraction_installed([hot, [(300, 0.9)]], 330), (1, 1))

print("one_run_false_accept and runs_needed")
check("false accept falls as f rises", acc.one_run_false_accept(0.8) < acc.one_run_false_accept(0.2), True)
check("f = 0 means the data never installs, no audit can be wrong", acc.runs_needed(0.0), None)
check("f = 1 needs a single run", acc.runs_needed(1.0), 1)
check("f = 0.2 needs 14 order re-draws at alpha 0.05", acc.runs_needed(0.2), 14)
check("f = 0.6 needs 4", acc.runs_needed(0.6), 4)
check("more runs are needed as f falls", acc.runs_needed(0.1) > acc.runs_needed(0.5), True)

print("separation: the deterministic headline form")
check("a separation window can be found", acc.separation([hot], [cold], G), [320, 330, 340])
check("the window shrinks when the cold arm installs late in it",
      acc.separation([hot], [cold[:-1] + [(340, 0.9)]], G), [320, 330])
check("no window when both sets agree", acc.separation([hot], [hot], G), [])
check("no window when the sets are the wrong way round", acc.separation([cold], [hot], G), [])
check("an empty set gives no window", acc.separation([], [cold], G), [])

print("clopper_pearson: exact interval, checked against published values")
for k, n, want in ((2, 9, (0.0281, 0.6000)), (6, 9, (0.2993, 0.9251)),
                   (0, 5, (0.0, 0.5218)), (5, 5, (0.4782, 1.0)), (1, 3, (0.0084, 0.9057))):
    got = acc.clopper_pearson(k, n)
    check(f"CP({k},{n})", tuple(round(g, 3) for g in got), tuple(round(w, 3) for w in want))
check("n = 0 gives no interval", acc.clopper_pearson(0, 0), (None, None))

print("\nrehearsal demos:", sum(ok), "/", len(ok))
sys.exit(0 if all(ok) else 1)
