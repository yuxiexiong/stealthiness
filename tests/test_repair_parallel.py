"""Real CPU subprocesses only; sentinel CUDA identities never access a GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "attribution-visualization/visual-evidence-repair"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))
from repair import parallel
from experiments import measure


WORKER = """
import json, os, signal, sys, time
from pathlib import Path
root, lane, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert sys.argv[4:] == ['--device', 'cuda:0']
def stop(signum, frame):
    (root / ('stopped-' + lane)).write_text(str(signum))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root / ('ready-' + lane)).write_text(json.dumps({
    'pid': os.getpid(), 'pgid': os.getpgrp(), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
    'time': time.monotonic()}))
deadline = time.monotonic() + 4
while not (root / ('ready-' + str(1-int(lane)))).exists():
    if time.monotonic() > deadline:
        raise RuntimeError('the other lane did not overlap')
    time.sleep(.01)
if mode == 'fail':
    raise SystemExit(7)
if mode == 'wait':
    while True:
        time.sleep(.05)
time.sleep(.1)
"""


class RepairParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.worker = self.root / "worker.py"
        self.worker.write_text(WORKER)

    def commands(self, modes):
        return [[[sys.executable, "-S", str(self.worker), str(self.root), str(lane), mode,
                  "--device", "cuda:0"],
                 [sys.executable, "-S", "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                  str(self.root / f"next-{lane}")]] for lane, mode in enumerate(modes)]

    def start(self, queues):
        path = self.root / "queues.json"
        path.write_text(json.dumps(queues))
        process = subprocess.Popen([sys.executable, "-S", "-m", "repair.parallel", "--queues", str(path),
                                    "--output", str(self.root / "result")], cwd=PROJECT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-left,GPU-right"),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        def cleanup():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        self.addCleanup(cleanup)
        return process

    def completed(self, process):
        out, err = process.communicate(timeout=12)
        self.assertFalse(err, err)
        record = json.loads((self.root / "result/run.json").read_text())
        self.assertEqual(json.loads(out), record)
        return record

    def assert_reaped(self, record):
        for queue in record["jobs"]:
            for job in queue:
                if "pid" in job:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(job["pid"], 0)

    def test_overlapping_lanes_isolated_gpu_identity_and_order(self):
        queues = self.commands(["complete", "complete"])
        process = self.start(queues)
        record = self.completed(process)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["queues"], queues)
        first = [lane[0] for lane in record["jobs"]]
        self.assertLess(max(job["started_seconds"] for job in first), min(job["finished_seconds"] for job in first))
        for lane, gpu in enumerate(["GPU-left", "GPU-right"]):
            ready = json.loads((self.root / f"ready-{lane}").read_text())
            self.assertEqual(ready["gpu"], gpu)
            self.assertEqual(ready["pgid"], process.pid)  # no new child session/group
            self.assertTrue((self.root / f"next-{lane}").exists())
            self.assertGreaterEqual(record["jobs"][lane][1]["started_seconds"], first[lane]["finished_seconds"])
        self.assert_reaped(record)

    def test_failure_stops_both_queues_and_reaps_peer(self):
        process = self.start(self.commands(["fail", "wait"]))
        record = self.completed(process)
        self.assertEqual(process.returncode, 1)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["jobs"][0][0]["exit_code"], 7)
        self.assertEqual(record["jobs"][1][0]["status"], "terminated")
        self.assertTrue((self.root / "stopped-1").exists())
        self.assertTrue(all(lane[1]["status"] == "skipped" for lane in record["jobs"]))
        self.assertFalse(any(self.root.glob("next-*")))
        self.assert_reaped(record)

    def test_one_outer_measurement_charges_both_cards(self):
        path = self.root / "queues.json"
        path.write_text(json.dumps(self.commands(["complete", "complete"])))
        command = [sys.executable, "-S", "-m", "repair.parallel", "--queues", str(path),
                   "--output", str(self.root / "result")]
        with patch.object(measure, "gpu_sample", return_value={"devices": []}):
            measured = measure.run(command, self.root / "measurement", PROJECT,
                                   gpus=("GPU-left", "GPU-right"), interval=.02, timeout=5)
        self.assertEqual(measured["status"], "completed")
        self.assertEqual(measured["allocated_gpu_hours"], measured["elapsed_seconds"] * 2 / 3600)
        self.assertTrue(measured["process_group_cleanup"]["group_disappeared"])
        record = json.loads((self.root / "result/run.json").read_text())
        self.assertEqual(record["status"], "completed")
        self.assertGreaterEqual(measured["elapsed_seconds"], record["elapsed_seconds"])
        self.assert_reaped(record)

    def test_sigterm_reaps_children_and_stops_pending_tasks(self):
        process = self.start(self.commands(["wait", "wait"]))
        deadline = time.monotonic() + 5
        while len(list(self.root.glob("ready-*"))) != 2:
            self.assertIsNone(process.poll())
            if time.monotonic() >= deadline:
                self.fail("workers did not become ready")
            time.sleep(.01)
        process.terminate()
        record = self.completed(process)
        self.assertEqual(process.returncode, 128 + signal.SIGTERM)
        self.assertEqual(record["status"], "interrupted")
        self.assertTrue(all((self.root / f"stopped-{lane}").exists() for lane in range(2)))
        self.assertFalse(any(self.root.glob("next-*")))
        self.assert_reaped(record)

    def test_rejects_invalid_queues_gpu_count_and_output_reuse(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-left,GPU-right"):
            for queues in ([], [[], [["cmd"]]], [[["cmd"]]], [[["cmd"]], [[1]]]):
                with self.subTest(queues=queues), self.assertRaises(ValueError):
                    parallel.run(queues, self.root / "unused")
            with self.assertRaises(FileExistsError):
                parallel.run([[["cmd"]], [["cmd"]]], self.root)
        for visible in ("", "0", "0,0", "0, 1", "0,1,2"):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=visible), self.assertRaises(ValueError):
                parallel.run([[["cmd"]], [["cmd"]]], self.root / "unused")


if __name__ == "__main__":
    unittest.main()
