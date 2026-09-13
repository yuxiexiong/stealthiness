"""Mock CPU launch contract: wait, technical smoke gate, separate GPU charges."""
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
    def test_live_controller_blocks_even_when_assigned_gpus_are_idle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ['editclevr-100161', 'editclevr-100669', 'a', 'b', 'c', 'd']
            write_json(root / 'manifest.json', {'count': 6, 'cases': [{'cluster_id': c} for c in ids]})
            write_json(root / 'config.json', {})
            controllers = [{'pid': 12345, 'start_ticks': '98765'}]
            write_json(root / 'after.json', controllers)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), \
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

    def test_wait_smoke_failure_barrier_and_null_scientific_results(self):
        for fail_smoke in (False, True):
            with self.subTest(fail_smoke=fail_smoke), tempfile.TemporaryDirectory() as directory:
                root, events = Path(directory), []
                ids = ['editclevr-100161', 'other-1', 'editclevr-100669', 'other-2', 'other-3', 'other-4']
                write_json(root / 'manifest.json', {'count': 6, 'cases': [{'cluster_id': c} for c in ids]})
                write_json(root / 'config.json', {})

                def measured(command, output, cwd, **kwargs):
                    phase = 'smoke' if '--smoke' in command else 'full'
                    events.append(phase)
                    self.assertNotIn('timeout', kwargs)
                    self.assertEqual(kwargs['gpus'], ['0'] if phase == 'smoke' else ['0', '1'])
                    self.assertEqual(command[0], sys.executable)
                    if phase == 'smoke':
                        targets = [(root / 'run/smoke', launcher.SMOKE_CASES)]
                    else:
                        queues = read_json(root / 'run/queues.json')
                        self.assertEqual(len(queues), 2)
                        self.assertTrue(all('--smoke' not in queue[0] for queue in queues))
                        targets = [(root / f'run/lane-{lane}', ids[lane::2]) for lane in range(2)]
                    for target, cases in targets:
                        write_json(target / 'run.json', {'status': 'completed_with_unresolved',
                                   'parameter_updates': 0, 'smoke': phase == 'smoke'})
                        node = {'controls': {'empty_replay_exact': True, 'full_visual_matches_normal': True,
                                            'self_copy_exact': True},
                                'conditions': {'a': {'status': 'failed' if fail_smoke else 'measured',
                                                     'correct': False}}}
                        write_json(target / 'report.json', {'cases': [{'cluster_id': c, 'nodes': [node, node]}
                                                                    for c in cases], 'cpu_test': False})
                    return {'status': 'completed', 'allocated_gpu_hours': .1 if phase == 'smoke' else .2}

                def report(command, **kwargs):
                    events.append('report')
                    self.assertIn('repair.visual_probe', command)
                    self.assertTrue(kwargs['check'])

                # A busy sample resets the confirmation; no GPU work while busy.
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), \
                        patch.object(launcher.subprocess, 'check_output', return_value='0, GPU-a\n1, GPU-b\n'), \
                        patch.object(launcher, 'gpu_idle', side_effect=[False, True, True, True, True, True, True]), \
                        patch.object(launcher.time, 'sleep') as sleep, \
                        patch.object(launcher, 'measure_run', side_effect=measured), \
                        patch.object(launcher.subprocess, 'run', side_effect=report):
                    argv = ['--config', str(root / 'config.json'), '--manifest', str(root / 'manifest.json'),
                            '--output', str(root / 'run')]
                    if fail_smoke:
                        with self.assertRaisesRegex(RuntimeError, 'smoke interface'):
                            launcher.main(argv)
                    else:
                        self.assertEqual(launcher.main(argv), 0)
                    self.assertTrue(all(call.args == (60,) for call in sleep.call_args_list))
                    self.assertEqual(sleep.call_count, 2 if fail_smoke else 3)
                self.assertEqual(events, ['smoke'] if fail_smoke else ['smoke', 'full', 'report'])
                state = read_json(root / 'run/state.json')
                self.assertEqual(state['status'], 'failed' if fail_smoke else 'completed')
                self.assertAlmostEqual(state['allocated_gpu_hours'], .1 if fail_smoke else .3)
                self.assertFalse(state['automatic_time_stop'])


if __name__ == '__main__':
    unittest.main()
