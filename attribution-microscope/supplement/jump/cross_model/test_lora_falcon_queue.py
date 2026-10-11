import unittest
from pathlib import Path
import falcon_queue


class FalconQueueTests(unittest.TestCase):
    def test_only_five_poison_runs_and_verified_dependencies(self):
        jobs = falcon_queue.build_jobs(Path('/code'), Path('/run'))
        formal = [j for j in jobs if '_falcon_s' in j['name']]
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len(formal), 5)
        self.assertEqual([j['name'].split('_s')[1].split('_')[0] for j in formal],
                         [str(s) for s in range(1001, 1006)])
        for job in formal:
            self.assertIn(falcon_queue.BARRIER, job['deps'])
            self.assertIn(jobs[1]['name'], job['deps'])
            self.assertIn('file:/run/falcon_pilot_passed.json', job['deps'])
            self.assertIn('--profile full', job['cmd'])
            self.assertNotIn('--arm clean', job['cmd'])
        self.assertEqual(jobs[-1]['deps'], [j['name'] for j in formal])
        self.assertIn(falcon_queue.BARRIER, jobs[0]['deps'])


if __name__ == '__main__':
    unittest.main()
