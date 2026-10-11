"""Regression checks for ordered publication and evidence-preserving merges."""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import uncertainty_followup as u


class UncertaintyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU tests need explicitly empty CUDA visibility')

    def manifest(self):
        return dict(runtimes={k: {'python': '/python-' + k} for k in ('qwen', 'llava')},
                    diagnostic_tasks=[dict(id='qvl_s1004', python='/python-qvl'),
                                      dict(id='falcon_s1001_r05', python='/python-falcon'),
                                      dict(id='sd_s1001_r01', python='/python-sd')])

    def test_all_successors_block_on_previous_and_cpu_acceptance(self):
        root = Path('/isolated')
        jobs = u.build_jobs(root, self.manifest())
        self.assertEqual(len(jobs), 16)
        self.assertIn('qwen_gpu_preflight', jobs[0]['name'])
        self.assertIn('qwen_s1010', jobs[5]['name'])
        self.assertIn('llava_s1010', jobs[10]['name'])
        for index, job in enumerate(jobs):
            expected = ['file:/isolated/cpu_tests_passed.json', 'file:/isolated/preparation_passed.json']
            self.assertEqual(job['deps'], ([jobs[index - 1]['name']] if index else []) + expected)
            self.assertNotIn('JOBQ_TEST', job['cmd'])
            self.assertEqual(job['cwd'], '/isolated/code')
        self.assertIn('CUDA_VISIBLE_DEVICES= ', jobs[-1]['cmd'])
        self.assertIn('CUDA_VISIBLE_DEVICES= ', jobs[6]['cmd'])
        self.assertNotIn('CUDA_VISIBLE_DEVICES=0', str(jobs))

    def test_gpu_guard_rejects_wrong_or_manual_worker(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'JOBQ_GPU': '1'}):
            with self.assertRaises(ValueError): u.gpu_guard()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1', 'JOBQ_GPU': ''}):
            with self.assertRaises(ValueError): u.gpu_guard()

    def test_merge_requires_accepted_identity_and_preserves_full_denominator(self):
        curves = [dict(model='Qwen3-8B', seed=1006, arm='poison', view='primary', probe_n=60,
                       full_training_budget=1250, points=[[0, 0], [20, .1], [1250, 1]])]
        v = dict(passed=True, probe_n=60, points=[[0, 0], [20, .1], [21, .7]])
        result = u.merge_points(curves, 'Qwen3-8B', 1006, v, 60)
        self.assertEqual(result['full_training_budget'], 1250)
        self.assertEqual(result['points'][-1], (1250, 1))
        with self.assertRaises(ValueError): u.merge_points(curves, 'Qwen3-8B', 1006, dict(v, passed=False), 60)
        with self.assertRaises(ValueError): u.merge_points(curves, 'Qwen3-8B', 1006, dict(v, points=[[20, .2]]), 60)

    def test_llava_reconstructed_zero_never_becomes_an_original_point(self):
        curves = [dict(model='LLaVA-1.5-7B', seed=1004, arm='poison', view='primary', probe_n=200,
                       full_training_budget=1250, points=[[20, 0], [1250, .99]])]
        result = u.merge_points(curves, 'LLaVA-1.5-7B', 1004,
                               dict(passed=True, probe_n=200, points=[[0, .1], [20, 0], [355, .115]]), 200)
        self.assertEqual(result['points'][0], (20, 0))
        self.assertIn('every20', result['measurement_note'])

    def test_unique_cost_parent_and_unknown_preserved(self):
        a = dict(cost_id='outer', wall_seconds=10, phase='replay')
        b = dict(cost_id='inner', parent_cost_id='outer', wall_seconds=7, phase='native')
        c = dict(cost_id='unknown', wall_seconds=None, phase='prior_failed_cpu')
        values = u.unique_costs([a, copy.deepcopy(a), b, c])
        result = u.cost_ledger(values)
        self.assertEqual(result['known_incremental_wall_seconds'], 10)
        self.assertEqual(result['unknown_receipts'], 1)
        with self.assertRaises(ValueError): u.unique_costs([a, dict(a, wall_seconds=11)])


if __name__ == '__main__':
    unittest.main()
