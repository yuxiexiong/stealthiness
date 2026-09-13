"""CPU contract: one GPU, full case coverage, smoke gate and GPU accounting."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.__main__ import read_json, write_json
from tools import run_visual_probe as launcher


class VisualProbeLauncherTest(unittest.TestCase):
    def test_available_card_does_not_wait_for_other_card_or_its_controller(self):
        controller = {'pid': 12345, 'start_ticks': '98765'}
        for blocked_by_controller in (False, True):
            with self.subTest(blocked_by_controller=blocked_by_controller):
                candidates = [{'gpu': 'GPU-a', 'after': [controller] if blocked_by_controller else []},
                              {'gpu': 'GPU-b', 'after': []}]
                queried = []

                def idle(uuids):
                    queried.append(uuids)
                    # An apparently idle card is still reserved by its live controller.
                    return blocked_by_controller or uuids == ['GPU-b']

                with patch.object(launcher, 'process_alive', return_value=True), \
                        patch.object(launcher, 'gpu_idle', side_effect=idle), \
                        patch.object(launcher.time, 'sleep') as sleep:
                    selected = launcher.wait_for_available(candidates, 'smoke', lambda **state: None)
                self.assertEqual(selected, candidates[1])
                sleep.assert_called_once_with(60)
                if blocked_by_controller:
                    self.assertNotIn(['GPU-a'], queried)
                self.assertEqual(queried.count(['GPU-b']), 3)  # Two samples and handoff recheck.

    def test_candidate_idle_checks_cannot_accumulate_across_cards(self):
        candidates = [{'gpu': 'GPU-a', 'after': []}, {'gpu': 'GPU-b', 'after': []}]
        samples = {'GPU-a': iter([True, False, True, True, True]),
                   'GPU-b': iter([False, True, False])}
        with patch.object(launcher, 'gpu_idle', side_effect=lambda uuids: next(samples[uuids[0]])), \
                patch.object(launcher.time, 'sleep') as sleep:
            selected = launcher.wait_for_available(candidates, 'smoke', lambda **state: None)
        self.assertEqual(selected, candidates[0])
        self.assertEqual(sleep.call_count, 3)

    def test_busy_handoff_requires_two_fresh_idle_checks(self):
        candidates = [{'gpu': 'GPU-a', 'after': []}]
        with patch.object(launcher, 'gpu_idle', side_effect=[True, True, False, True, True, True]) as idle, \
                patch.object(launcher.time, 'sleep') as sleep:
            selected = launcher.wait_for_available(candidates, 'smoke', lambda **state: None)
        self.assertEqual(selected, candidates[0])
        self.assertEqual(idle.call_count, 6)
        self.assertEqual(sleep.call_count, 3)

    def test_live_controller_blocks_even_when_assigned_gpu_is_idle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ['editclevr-100161', 'editclevr-100669', 'a', 'b', 'c', 'd']
            write_json(root / 'manifest.json', {'count': 6, 'cases': [{'cluster_id': c} for c in ids]})
            write_json(root / 'config.json', {})
            controllers = [{'pid': 12345, 'start_ticks': '98765'}]
            write_json(root / 'after.json', controllers)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-b'}), \
                    patch.object(launcher.subprocess, 'check_output', return_value='0, GPU-a\n1, GPU-b\n'), \
                    patch.object(launcher, 'process_alive', return_value=True) as alive, \
                    patch.object(launcher, 'gpu_idle', return_value=True) as idle, \
                    patch.object(launcher, 'measure_run') as measured, \
                    patch.object(launcher.time, 'sleep', side_effect=RuntimeError('end CPU waiting test')) as sleep:
                with self.assertRaisesRegex(RuntimeError, 'end CPU waiting test'):
                    launcher.main(['--config', str(root / 'config.json'), '--manifest', str(root / 'manifest.json'),
                                   '--after', str(root / 'after.json'), '--output', str(root / 'run')])
                alive.assert_called_once_with(controllers[0])
                idle.assert_not_called()
                measured.assert_not_called()
                sleep.assert_called_once_with(60)
            state = read_json(root / 'run/state.json')
            self.assertEqual(state['phase'], 'waiting_smoke')
            self.assertEqual(state['controllers'], controllers)
            self.assertEqual(state['idle_checks'], 0)
            self.assertEqual(state['allocated_gpu_hours'], 0)
            self.assertEqual(read_json(root / 'run/launch.json')['after'], controllers)

    def test_single_gpu_full_coverage_smoke_barrier_and_null_scientific_results(self):
        for failure, choose_available in ((None, False), (None, True), ('smoke', False),
                                          ('missing_full_case', False)):
            with self.subTest(failure=failure, choose_available=choose_available), \
                    tempfile.TemporaryDirectory() as directory:
                root, events = Path(directory).resolve(), []
                ids = ['editclevr-100161', 'other-1', 'editclevr-100669', 'other-2', 'other-3', 'other-4']
                write_json(root / 'manifest.json', {'count': 6, 'cases': [{'cluster_id': c} for c in ids]})
                write_json(root / 'config.json', {})
                candidates = [{'gpu': 'GPU-a', 'after': []}, {'gpu': 'GPU-b', 'after': []}]
                write_json(root / 'candidates.json', candidates)
                assigned = ['GPU-b'] if choose_available else ['1']

                def measured(command, output, cwd, **kwargs):
                    phase = 'smoke' if '--smoke' in command else 'full'
                    events.append(phase)
                    self.assertNotIn('timeout', kwargs)
                    self.assertEqual(kwargs['gpus'], assigned)
                    if choose_available:
                        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-b')
                    self.assertEqual(command[0], sys.executable)
                    self.assertEqual(command[1:4], ['-m', 'repair.visual_probe', 'map'])
                    self.assertEqual(command[command.index('--device') + 1], 'cuda:0')
                    if phase == 'smoke':
                        targets = [(root / 'run/smoke', launcher.SMOKE_CASES)]
                    else:
                        self.assertEqual(command[command.index('--lanes') + 1], '1')
                        self.assertEqual(command[command.index('--lane-index') + 1], '0')
                        self.assertEqual(command[command.index('--output') + 1], str(root / 'run/full'))
                        self.assertFalse((root / 'run/queues.json').exists())
                        targets = [(root / 'run/full', ids[:-1] if failure == 'missing_full_case' else ids)]
                    for target, cases in targets:
                        write_json(target / 'run.json', {'status': 'completed_with_unresolved',
                                   'parameter_updates': 0, 'smoke': phase == 'smoke'})
                        node = {'controls': {'empty_replay_exact': True, 'full_visual_matches_normal': True,
                                            'self_copy_exact': True},
                                'conditions': {'a': {'status': 'failed' if failure == 'smoke' else 'measured',
                                                     'correct': False}}}
                        write_json(target / 'report.json', {'cases': [{'cluster_id': c, 'nodes': [node, node]}
                                                                    for c in cases], 'cpu_test': False})
                    return {'status': 'completed', 'allocated_gpu_hours': .1 if phase == 'smoke' else .2}

                def report(command, **kwargs):
                    events.append('report')
                    self.assertIn('repair.visual_probe', command)
                    self.assertEqual(command[command.index('--runs') + 1:command.index('--output')],
                                     [str(root / 'run/full')])
                    self.assertTrue(kwargs['check'])

                # A busy sample resets the confirmation; no GPU work while busy.
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), \
                        patch.object(launcher.subprocess, 'check_output', return_value='0, GPU-a\n1, GPU-b\n'), \
                        patch.object(launcher, 'gpu_idle', side_effect=[False, True, True, True, True, True, True]) as idle, \
                        patch.object(launcher.time, 'sleep') as sleep, \
                        patch.object(launcher, 'wait_for_available', return_value=candidates[1]) as select, \
                        patch.object(launcher, 'measure_run', side_effect=measured), \
                        patch.object(launcher.subprocess, 'run', side_effect=report):
                    argv = ['--config', str(root / 'config.json'), '--manifest', str(root / 'manifest.json'),
                            '--output', str(root / 'run')]
                    if choose_available:
                        argv += ['--gpu-candidates', str(root / 'candidates.json')]
                    if failure:
                        message = 'smoke interface' if failure == 'smoke' else 'full did not complete'
                        with self.assertRaisesRegex(RuntimeError, message):
                            launcher.main(argv)
                    else:
                        self.assertEqual(launcher.main(argv), 0)
                    self.assertTrue(all(call.args == (60,) for call in sleep.call_args_list))
                    self.assertEqual(sleep.call_count, 2 if failure == 'smoke' or choose_available else 3)
                    self.assertTrue(all(call.args == (['GPU-b'],) for call in idle.call_args_list))
                    if choose_available:
                        select.assert_called_once()
                        self.assertEqual(select.call_args.args[:2], (candidates, 'smoke'))
                    else:
                        select.assert_not_called()
                expected_events = ['smoke'] if failure == 'smoke' else ['smoke', 'full']
                self.assertEqual(events, expected_events if failure else expected_events + ['report'])
                state = read_json(root / 'run/state.json')
                self.assertEqual(state['status'], 'failed' if failure else 'completed')
                self.assertEqual(state['assigned_gpus'], assigned)
                self.assertAlmostEqual(state['allocated_gpu_hours'], .1 if failure == 'smoke' else .3)
                self.assertFalse(state['automatic_time_stop'])

    def test_rejects_multi_gpu_assignment_and_unknown_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), \
                    patch.object(launcher, 'measure_run') as measured, \
                    patch.object(launcher.subprocess, 'check_output') as query:
                with self.assertRaises(SystemExit):
                    launcher.main(['--config', str(root / 'config.json'), '--manifest', str(root / 'manifest.json'),
                                   '--output', str(root / 'run')])
                measured.assert_not_called()
                query.assert_not_called()
                self.assertFalse((root / 'run').exists())
            with patch.object(launcher.subprocess, 'check_output', return_value='0, GPU-a\n1, GPU-b\n'):
                with self.assertRaises(ValueError):
                    launcher.gpu_uuids(['99'])


if __name__ == '__main__':
    unittest.main()
