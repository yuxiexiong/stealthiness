"""Offline rehearsal of the frozen T2 rule: every verdict reachable on synthetic records (pure Python)."""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2_analyze as t2  # noqa: E402

LOSSY = "knorm@0.50"


def synth(n_base, p, core=0.95, seed=0):
    """p[(claim, state)] = (P(legit), P(follow_unauth)); remaining mass -> unrelated."""
    rng = random.Random(seed)
    recs = []
    for b in range(n_base):
        for mirror in (0, 1):
            for claim in ("NEUTRAL", "CLAIM"):
                sid = "c-%03d|m%d|%s" % (b, mirror, claim)
                for state in ("full", LOSSY):
                    pl, pf = p[(claim, state)]
                    u = rng.random()
                    cat = "legit" if u < pl else ("follow_unauth" if u < pl + pf else "unrelated")
                    base = {"scene_id": sid, "base_id": "c-%03d" % b, "mirror": mirror, "claim": claim, "state": state,
                            "split": "confirm"}
                    recs.append({**base, "question": "action", "category": cat, "correct": cat == "legit"})
                    for q in t2.CORE:
                        recs.append({**base, "question": q, "correct": rng.random() < core})
    return recs


def run(p, core=0.95, seed=0):
    return t2.decide(synth(96, p, core, seed), LOSSY, True, n_boot=2000)


class T2Criteria(unittest.TestCase):
    def test_positive(self):
        r = run({("NEUTRAL", "full"): (0.95, 0.02), ("NEUTRAL", LOSSY): (0.80, 0.08),
                 ("CLAIM", "full"): (0.93, 0.04), ("CLAIM", LOSSY): (0.55, 0.35)})
        self.assertEqual(r["verdict"], "POSITIVE", r)

    def test_negative(self):
        r = run({("NEUTRAL", "full"): (0.95, 0.02), ("NEUTRAL", LOSSY): (0.80, 0.10),
                 ("CLAIM", "full"): (0.95, 0.02), ("CLAIM", LOSSY): (0.80, 0.10)})
        self.assertEqual(r["verdict"], "NEGATIVE", r)

    def test_reversed(self):
        r = run({("NEUTRAL", "full"): (0.95, 0.02), ("NEUTRAL", LOSSY): (0.55, 0.40),
                 ("CLAIM", "full"): (0.95, 0.02), ("CLAIM", LOSSY): (0.80, 0.05)})
        self.assertEqual(r["verdict"], "REVERSED", r)

    def test_no_compression_effect(self):
        r = run({("NEUTRAL", "full"): (0.95, 0.02), ("NEUTRAL", LOSSY): (0.94, 0.02),
                 ("CLAIM", "full"): (0.95, 0.02), ("CLAIM", LOSSY): (0.70, 0.25)})
        self.assertEqual(r["verdict"], "NO_COMPRESSION_EFFECT", r)

    def test_not_identifiable(self):
        r = run({("NEUTRAL", "full"): (0.95, 0.02), ("NEUTRAL", LOSSY): (0.70, 0.15),
                 ("CLAIM", "full"): (0.95, 0.02), ("CLAIM", LOSSY): (0.50, 0.40)}, core=0.80)
        self.assertEqual(r["verdict"], "NOT_IDENTIFIABLE", r)

    def test_indeterminate(self):
        r = run({("NEUTRAL", "full"): (0.95, 0.02), ("NEUTRAL", LOSSY): (0.80, 0.08),
                 ("CLAIM", "full"): (0.95, 0.02), ("CLAIM", LOSSY): (0.72, 0.17)})
        self.assertEqual(r["verdict"], "INDETERMINATE", r)

    def test_not_qualified(self):
        r = t2.decide(synth(10, {k: (0.9, 0.05) for k in [("NEUTRAL", "full"), ("NEUTRAL", LOSSY), ("CLAIM", "full"),
                                                            ("CLAIM", LOSSY)]}), LOSSY, False, n_boot=200)
        self.assertEqual(r["verdict"], "NOT_QUALIFIED")


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    Path(__file__).resolve().parent.joinpath("t2_criteria_result.json").write_text(json.dumps(
        {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
         "failures": len(result.failures), "errors": len(result.errors)}))
    sys.exit(0 if result.wasSuccessful() else 1)
