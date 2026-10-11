import unittest
from summarize import summarize


class Crossings(unittest.TestCase):
    def test_censoring_and_rebound(self):
        s = summarize([(0, 0.0), (2, 0.2), (4, 0.95), (6, 0.5), (8, 0.96)])
        self.assertEqual(s["sampled_width"], 2)
        self.assertEqual(s["t90"]["interval"], [2, 4])
        self.assertEqual(s["sustained_t90"]["step"], 8)
        self.assertEqual(summarize([(0, .15), (2, .4)])["t10"]["status"], "left_censored")
        self.assertIsNone(summarize([(0, 0), (2, .5)])["t90"]["step"])
        for points in [[], [(0, float("nan"))], [(0, .1), (0, .2)]]:
            with self.assertRaises(ValueError):
                summarize(points)


if __name__ == "__main__":
    unittest.main()
