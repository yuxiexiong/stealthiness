from pathlib import Path
import unittest
import classification_queue as queue


class QueueTests(unittest.TestCase):
    def test_frozen_eight_arms_and_full_barrier(self):
        jobs=queue.build_jobs(Path('/run'))
        formal=[x for x in jobs if '--phase formal' in x['cmd']]
        self.assertEqual(len(formal),8)
        self.assertEqual(sum('--arm poison' in x['cmd'] for x in formal),6)
        self.assertEqual(sum('--arm clean' in x['cmd'] for x in formal),2)
        self.assertEqual(len(jobs),14)
        self.assertIn(queue.BARRIER,jobs[0]['deps'])
        for job in formal:
            self.assertIn(queue.BARRIER,job['deps'])
            self.assertTrue(any('_warmup' in d for d in job['deps']))
            self.assertTrue(any('_pilot_passed.json' in d for d in job['deps']))
        self.assertEqual(jobs[-1]['deps'],[x['name'] for x in formal])
        self.assertEqual(len({x['name'] for x in jobs}),len(jobs))


if __name__ == '__main__':
    unittest.main()
