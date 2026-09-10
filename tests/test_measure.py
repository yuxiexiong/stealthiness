"""Real CPU process groups; GPU telemetry is replaced and no GPU is queried."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import measure


@unittest.skipUnless(os.name == "posix", "The measurement runner uses POSIX process groups")
class MeasureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sample = patch.object(measure, "gpu_sample", side_effect=lambda gpus: {"devices": []})
        self.sample.start()
        self.addCleanup(self.sample.stop)

    def run_command(self, code, **kwargs):
        return measure.run([sys.executable, "-c", code], self.root / "measurement",
                           self.root, gpus=("cpu-fixture-0", "cpu-fixture-1"), interval=.02, **kwargs)

    def orphan_command(self, ignore_term=False, keep_leader=False):
        worker = self.root / "worker.py"
        worker.write_text(
            "import json,os,signal,time\nfrom pathlib import Path\n"
            "def terminate(signum, frame):\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "    time.sleep(.3)\n"
            "    Path('terminated').write_text(str(time.monotonic()))\n"
            "    raise SystemExit(0)\n"
            f"signal.signal(signal.SIGTERM, {'signal.SIG_IGN' if ignore_term else 'terminate'})\n"
            "Path('worker.json').write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}))\n"
            "while True:\n"
            "    Path('heartbeat').write_text(str(time.monotonic()))\n"
            "    time.sleep(.02)\n")
        # The child inherits the leader's private group; no subprocess detaches.
        return (
            "import subprocess,sys,time\nfrom pathlib import Path\n"
            f"subprocess.Popen([sys.executable, {str(worker)!r}])\n"
            "deadline=time.monotonic()+5\n"
            "while not Path('worker.json').exists():\n"
            "    if time.monotonic()>deadline: raise SystemExit(2)\n"
            "    time.sleep(.01)\n"
            "Path('leader-exit').write_text(str(time.monotonic()))\n"
            + ("time.sleep(60)\n" if keep_leader else ""))

    def assert_worker_stopped(self):
        worker = json.loads((self.root / "worker.json").read_text())
        self.assertNotEqual(worker["pgid"], os.getpgrp())
        heartbeat = (self.root / "heartbeat").read_bytes()
        time.sleep(.15)
        self.assertEqual((self.root / "heartbeat").read_bytes(), heartbeat)

    def tearDown(self):
        path = self.root / "worker.json"
        if path.exists():
            pgid = json.loads(path.read_text())["pgid"]
            if pgid != os.getpgrp():
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_normal_exit_has_no_cleanup_sleep_or_signal(self):
        result = self.run_command("pass")
        self.assertEqual((result["status"], result["exit_code"]), ("completed", 0))
        self.assertEqual(result["process_group_cleanup"]["signals"], [])
        self.assertTrue(result["process_group_cleanup"]["group_disappeared"])
        self.assertLess(result["elapsed_seconds"], 5)

    def test_leader_exit_cleans_descendant_and_charges_termination_time(self):
        result = self.run_command(self.orphan_command(), timeout=10)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn(result["status"], ("interrupted", "cleanup_failed"))
        self.assertIn("SIGTERM", result["process_group_cleanup"]["signals"])
        terminated = float((self.root / "terminated").read_text())
        leader_exit = float((self.root / "leader-exit").read_text())
        self.assertGreater(terminated - leader_exit, .25)
        self.assertGreaterEqual(result["elapsed_seconds"], terminated - leader_exit)
        self.assertEqual(result["allocated_gpu_hours"], result["elapsed_seconds"] * 2 / 3600)
        self.assertEqual(json.loads((self.root / "measurement/run.json").read_text()), result)
        self.assert_worker_stopped()

    def test_timeout_kills_ignoring_descendant_after_leader_dies(self):
        result = self.run_command(self.orphan_command(ignore_term=True, keep_leader=True), timeout=.4)
        self.assertIn(result["status"], ("timeout", "cleanup_failed"))
        self.assertEqual(result["exit_code"], -signal.SIGTERM)
        self.assertEqual(result["process_group_cleanup"]["signals"], ["SIGTERM", "SIGKILL"])
        self.assertGreaterEqual(result["elapsed_seconds"], 5)
        self.assert_worker_stopped()

    def test_remaining_group_is_unconfirmed_after_bounded_cleanup(self):
        child = Mock(pid=987654321)
        with patch.object(measure.os, "killpg") as killpg, \
             patch.object(measure.time, "monotonic", side_effect=[0, 6, 7, 13]):
            result = measure.cleanup_group(child)
        self.assertFalse(result["group_disappeared"])
        self.assertEqual(result["signals"], ["SIGTERM", "SIGKILL"])
        self.assertEqual([call.args for call in killpg.call_args_list],
                         [(child.pid, 0), (child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)])

    def test_permission_error_cannot_be_recorded_as_group_disappearance(self):
        with patch.object(measure.os, "killpg", side_effect=PermissionError("CPU fixture denial")):
            result = self.run_command("pass")
        self.assertEqual((result["status"], result["exit_code"]), ("cleanup_failed", 0))
        self.assertFalse(result["process_group_cleanup"]["group_disappeared"])
        self.assertIn("CPU fixture denial", result["error"])


if __name__ == "__main__":
    unittest.main()
