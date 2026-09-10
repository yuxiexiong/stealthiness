"""Waiter checks use mocked telemetry and a harmless CPU child, never CUDA."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1] / 'attribution-visualization/visual-evidence-repair'
sys.path.insert(0, str(PROJECT))
from repair import server_queue as queue


class ServerQueueTests(unittest.TestCase):
    def test_controller_then_busy_then_two_idle_checks_before_cpu_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = root / 'launch.json'
            marker = root / 'child.txt'
            launch.write_text(json.dumps({'ready': True, 'cwd': str(root), 'required_files': [str(launch)],
                'command': [sys.executable, '-c', 'from pathlib import Path; Path("child.txt").write_text("ok")']}))
            config = {'after': [{'pid': 123, 'start_ticks': 456}], 'gpus': ['a', 'b'], 'launch_manifest': str(launch)}
            with patch.object(queue, 'process_alive', side_effect=[True, False, False, False, False]), \
                 patch.object(queue, 'gpu_idle', side_effect=[False, True, True, True]) as idle, \
                 patch.object(queue.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(queue.wait_and_run(root, config), 0)
            self.assertEqual(sleep.call_count, 3)
            self.assertEqual(idle.call_count, 4)
            self.assertEqual(marker.read_text(), 'ok')
            self.assertEqual(json.loads((root / 'state.json').read_text())['status'], 'completed')

    def test_no_launch_without_inputs_and_pid_reuse_does_not_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(queue.launch_at(root / 'missing.json')[0])
            data = root / 'launch.json'
            data.write_text(json.dumps({'ready': False}))
            self.assertIsNone(queue.launch_at(data)[0])
        fields = ['S', '1'] + ['0'] * 17 + ['999']
        with patch.object(Path, 'read_text', return_value='123 (controller with spaces) ' + ' '.join(fields)):
            self.assertFalse(queue.process_alive({'pid': 123, 'start_ticks': 998}))
            self.assertTrue(queue.process_alive({'pid': 123, 'start_ticks': 999}))

    def test_missing_gpu_and_unknown_telemetry_never_count_as_idle(self):
        with patch.object(queue.subprocess, 'check_output', side_effect=['a, 9\n', 'unused']):
            self.assertFalse(queue.gpu_idle(['a', 'b']))
        with patch.object(queue.subprocess, 'check_output', side_effect=['', 'a, 0, 0\n']):
            self.assertFalse(queue.gpu_idle(['a', 'b']))
        with patch.object(queue.subprocess, 'check_output', side_effect=['', 'a, N/A, 0\nb, 0, 0\n']):
            with self.assertRaises(ValueError):
                queue.gpu_idle(['a', 'b'])


if __name__ == '__main__':
    unittest.main()
