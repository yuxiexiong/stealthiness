import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jump_v1 import summarize
from lora_llm import replay_limit, learning_rate


class JumpTests(unittest.TestCase):
    def test_percentage_points_not_relative_growth(self):
        self.assertEqual(summarize([(0, .01), (20, .02)])['status'], 'not_observed_at_available_resolution')
        self.assertEqual(summarize([(0, .1), (20, .6)])['status'], 'observed')

    def test_exact_boundaries_and_no_ninety_percent_requirement(self):
        for pts in [[(0, 0), (20, .5)], [(0, .4), (10, .9)]]:
            self.assertEqual(summarize(pts)['status'], 'observed')
        for pts in [[(0, 0), (21, .5)], [(0, 0), (20, .49)], [(0, 0), (200, 1)]]:
            self.assertEqual(summarize(pts)['status'], 'not_observed_at_available_resolution')

    def test_scan_all_windows_not_only_adjacent_or_initial(self):
        s = summarize([(0, 0), (5, .2), (10, .3), (15, .6), (20, .2), (25, .75)])
        self.assertEqual(s['first_window']['end'], 15)
        self.assertEqual(s['fastest_observed_window']['updates'], 5)
        self.assertAlmostEqual(s['largest_later_drop_pp'], 40)

    def test_sparse_curve_does_not_invent_width(self):
        s = summarize([(0, 0), (100, .9)])
        self.assertIsNone(s['first_window'])
        self.assertIsNone(s['maximum_20_update_window'])
        self.assertEqual(s['maximum_sampling_gap'], 100)

    def test_refuse_invalid_or_conflicting_measurements(self):
        for pts in [[], [(0, math.nan)], [(0, 1.1)], [(-1, 0)], [(20, 0), (0, 1)], [(0, 0), (0, 1)]]:
            with self.assertRaises(ValueError): summarize(pts)

    def test_original_runs_unchanged_and_replay_keeps_original_lr_schedule(self):
        self.assertEqual(replay_limit(SimpleNamespace(replay_stop_after=None), 1250), 1250)
        a = SimpleNamespace(replay_stop_after=120, profile='full', replay_anchors='original',
                            dense_start=80, dense_end=100)
        self.assertEqual(replay_limit(a, 1250), 120)
        self.assertNotEqual(learning_rate(100, total=1250), learning_rate(100, total=120))
        for changes in [{'profile': 'pilot'}, {'replay_anchors': None}, {'replay_stop_after': 90},
                        {'replay_stop_after': 1251}, {'dense_start': None}, {'dense_end': None}]:
            with self.assertRaises(ValueError): replay_limit(SimpleNamespace(**{**vars(a), **changes}), 1250)

    def test_gpu0_is_rejected_before_any_tensor_allocation(self):
        from jump_v1_queue import guard_gpu
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'JOBQ_GPU': '0'}):
            with self.assertRaisesRegex(ValueError, 'GPU1'): guard_gpu()


if __name__ == '__main__': unittest.main()
