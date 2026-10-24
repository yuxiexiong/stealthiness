"""CONTRACT_S3 offline rehearsal. Each criterion demonstrated BOTH ways plus the
degenerate branches that must NOT read as a negative result. Exit 0 only if all pass."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import s3
# "s3" is also a DIRECTORY name: running this from the parent directory makes
# `import s3` bind the namespace package instead of s3/s3.py. Fail loudly.
assert os.path.dirname(os.path.abspath(s3.__file__)) == _HERE, (
    "wrong s3 module bound (%s); run this script from its own directory" % s3.__file__)

ok = []


def check(name, got, want):
    good = got == want
    ok.append(good)
    print(("  PASS  " if good else "  FAIL  ") + name + f"   got={got} want={want}")


HOT, COLD = [0.9, 0.85, 0.95], [0.1, 0.0, 0.2]
MIXED = [0.9, 0.1, 0.8]

print("primary criterion: clean separation at the budget")
check("can be met", s3.separated(HOT, COLD)["separated"], True)
check("can fail when the slow order also installs", s3.separated(HOT, HOT)["separated"], False)
check("can fail when the fast order does not install", s3.separated(COLD, COLD)["separated"], False)
check("not met when replication is short", s3.separated(HOT[:2], COLD)["separated"], False)
check("short replication is flagged, not silently negative",
      s3.separated(HOT[:2], COLD)["fully_replicated"], False)

print("the two degenerate branches must be named, not read as a negative")
check("all installed is flagged", s3.separated(HOT, HOT)["degenerate_all_installed"], True)
check("none installed is flagged", s3.separated(COLD, COLD)["degenerate_none_installed"], True)
check("a real separation is not flagged degenerate",
      (s3.separated(HOT, COLD)["degenerate_all_installed"],
       s3.separated(HOT, COLD)["degenerate_none_installed"]), (False, False))

print("control: initialisation replicates must agree within an order")
check("agreement can hold", s3.inits_agree(HOT), True)
check("agreement can fail", s3.inits_agree(MIXED), False)
check("agreement holds for an all-cold order", s3.inits_agree(COLD), True)
check("no replicates gives no reading, not agreement", s3.inits_agree([]), None)

print("missing readings propagate as None, never as not-installed")
check("None is not False", s3.installed(None), None)
check("a missing replicate leaves both counts", s3.arm_counts([0.9, None, 0.95]), (2, 2))

print("baseline side check")
check("same side reproduces", s3.baseline_reproduced(0.30, 0.215), True)
check("crossing the threshold does not", s3.baseline_reproduced(0.80, 0.215), False)
check("installed baseline reproduces", s3.baseline_reproduced(0.90, 0.96), True)
check("missing observation gives no reading", s3.baseline_reproduced(None, 0.215), None)

print("tie back to the prescription")
check("false accept with a 3-vs-3 separation is 0.5",
      round(s3.false_accept_if_one_run(HOT, COLD), 4), 0.5)
check("no arms gives no number", s3.false_accept_if_one_run([], []), None)

print("\nrehearsal demos:", sum(ok), "/", len(ok))
sys.exit(0 if all(ok) else 1)
