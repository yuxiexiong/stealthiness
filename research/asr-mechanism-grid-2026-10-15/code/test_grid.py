import unittest

import grid_common as gc_


def grid(fork_row, mature_row, other=None):
    hits = {}
    for m in gc_.MATURITY:
        row = fork_row if m == "fork" else (mature_row if m == gc_.PRIMARY_MATURE else (other or mature_row))
        for a, h in zip(gc_.ALPHAS, row):
            hits[gc_.cell(m, a)] = h
    return hits


class Classify(unittest.TestCase):
    def test_multiplicative(self):
        hits = grid(fork_row=(0, 0, 1, 2), mature_row=(1, 45, 58, 60))
        control = {gc_.cell("tj40", 1.0): 0, gc_.cell("tj40", 2.0): 2}
        self.assertEqual(gc_.classify_mother(hits, control), "multiplicative")

    def test_additive_by_strong_fork(self):
        hits = grid(fork_row=(0, 20, 40, 58), mature_row=(1, 22, 41, 59))
        control = {gc_.cell("tj40", 1.0): 0, gc_.cell("tj40", 2.0): 1}
        self.assertEqual(gc_.classify_mother(hits, control), "additive")

    def test_not_specific_when_random_direction_installs(self):
        hits = grid(fork_row=(0, 0, 1, 2), mature_row=(1, 45, 58, 60))
        control = {gc_.cell("tj40", 1.0): 10, gc_.cell("tj40", 2.0): 40}
        self.assertEqual(gc_.classify_mother(hits, control), "not_specific")

    def test_cannot_measure_cases(self):
        control = {gc_.cell("tj40", 1.0): 0, gc_.cell("tj40", 2.0): 0}
        self.assertEqual(gc_.classify_mother(grid((0, 0, 0, 0), (10, 40, 55, 60)), control), "cannot_measure_B2_installs_alone")
        self.assertEqual(gc_.classify_mother(grid((0, 0, 1, 2), (0, 3, 8, 20)), control), "cannot_measure_no_cell_on")

    def test_unclear_and_judge(self):
        control = {gc_.cell("tj40", 1.0): 0, gc_.cell("tj40", 2.0): 0}
        self.assertEqual(gc_.classify_mother(grid((0, 5, 12, 20), (0, 10, 25, 50)), control), "unclear")
        self.assertEqual(gc_.judge({"s1": "multiplicative", "s2": "multiplicative", "s3": "multiplicative"})["tier"], "multiplicative")
        self.assertEqual(gc_.judge({"s1": "multiplicative", "s2": "additive", "s3": "multiplicative"})["tier"], "inconsistent")


if __name__ == "__main__":
    unittest.main()
