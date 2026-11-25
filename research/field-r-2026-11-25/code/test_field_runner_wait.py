"""Pure test of field_runner.wait_gpu_free (no GPU): busy readings reset the streak; failed queries count as busy."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import field_runner as fr


def run(readings):
    it = iter(readings)
    def query(gpu):
        v = next(it)
        if v == "err":
            raise RuntimeError("nvidia-smi failed")
        return v
    sleeps = []
    checks = fr.wait_gpu_free(0, query=query, sleep=sleeps.append)
    return checks, len(sleeps)


class Wait(unittest.TestCase):
    def test_free_immediately(self):
        self.assertEqual(run([0, 0, 0]), (3, 2))

    def test_busy_resets_streak(self):
        self.assertEqual(run([0, 0, 90000, 0, 0, 0]), (6, 5))

    def test_threshold_is_strict(self):
        self.assertEqual(run([2000, 1999, 1999, 1999]), (4, 3))

    def test_query_failure_counts_as_busy(self):
        self.assertEqual(run(["err", 0, 0, 0]), (4, 3))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=1).result.wasSuccessful() else 1)
