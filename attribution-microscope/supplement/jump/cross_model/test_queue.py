import json
import tempfile
import unittest
from pathlib import Path
from queue_runs import build_jobs, publish


class QueueTests(unittest.TestCase):
    def test_dependency_graph_and_idempotent_publish(self):
        jobs = build_jobs(Path('/code with spaces'), Path('/run'), Path('/python'))
        names = {j['name'] for j in jobs}
        self.assertEqual(len(names), len(jobs))
        self.assertEqual(len(jobs), 22)
        seen = set()
        for job in jobs:
            self.assertTrue(all(d.startswith('file:') or d in seen for d in job['deps']))
            seen.add(job['name'])
            self.assertIn('HF_HUB_OFFLINE=1', job['cmd'])
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            publish(jobs, queue)
            publish(jobs, queue)
            self.assertEqual(len(list((queue / 'jobs').glob('*.json'))), len(jobs))
            changed = dict(jobs[-1], cmd='changed')
            with self.assertRaises(ValueError):
                publish([changed], queue)


if __name__ == '__main__':
    unittest.main()
