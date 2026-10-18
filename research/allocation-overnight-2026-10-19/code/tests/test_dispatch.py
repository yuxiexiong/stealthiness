"""CPU-only checks of accounting, claims, and isolated child failures."""
import importlib.util
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "dispatch.py"
spec = importlib.util.spec_from_file_location("overnight_dispatch", MODULE)
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def concurrent_claim(root, contract, lane, output):
    output.put(dispatch.claim(Path(root), contract, lane)["name"])


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = {"runtime": {"python": sys.executable}, "jobs": [
            {"name": "first", "est_minutes": 1}, {"name": "second", "est_minutes": 1}]}
        self.now = time.time()

    def state(self):
        return json.loads((self.root / "budget.json").read_text())

    def test_two_processes_claim_in_order_without_duplicates(self):
        ctx = multiprocessing.get_context("fork")
        output = ctx.Queue()
        children = [ctx.Process(target=concurrent_claim, args=(
            str(self.root), self.contract, str(i), output)) for i in range(2)]
        for child in children:
            child.start()
        for child in children:
            child.join(5)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual({output.get(timeout=1), output.get(timeout=1)}, {"first", "second"})
        self.assertEqual(sum(r["reserved_seconds"] for r in self.state()["jobs"].values()), 120)

    def test_insufficient_total_budget_defers_without_start(self):
        with patch.object(dispatch, "LIMIT_SECONDS", 40):
            self.assertIsNone(dispatch.claim(self.root, self.contract, "0", self.now))
        self.assertEqual({r["status"] for r in self.state()["jobs"].values()}, {"deferred"})

    def test_after_former_cutoff_still_claims_and_runs_without_a_new_deadline(self):
        former_cutoff = datetime.fromisoformat("2026-10-19T10:51:14+00:00").timestamp()
        (self.root / "overnight_run.py").write_text("print('completed after former cutoff')\n")
        with patch.object(dispatch, "__file__", str(self.root / "dispatch.py")), \
             patch.object(dispatch.time, "time", return_value=former_cutoff + 86400):
            job = dispatch.claim(self.root, self.contract, "0")
            dispatch.run(self.root, self.contract, job, [False])
        self.assertEqual(self.state()["jobs"]["first"]["status"], "done")
        self.assertGreater(self.state()["jobs"]["first"]["start_epoch"], former_cutoff)
        self.assertIsNone(self.state()["deadline_epoch"])

    def test_caps_are_reserved_and_actual_wall_replaces_ceiling(self):
        with patch.object(dispatch, "LIMIT_SECONDS", 150):
            dispatch.claim(self.root, self.contract, "0", self.now)
            self.assertIsNone(dispatch.claim(self.root, self.contract, "1", self.now))
            self.assertEqual(self.state()["jobs"]["second"]["status"], "queued")
            dispatch.finish(self.root, self.contract, "first", {"status": "failed", "wall_seconds": 20})
            second = dispatch.claim(self.root, self.contract, "1", self.now)
            self.assertEqual(second["name"], "second")
            self.assertLessEqual(second["cap_seconds"] + 20, 150)
            self.assertEqual(self.state()["actual_seconds"], 20)
            self.assertIsNone(dispatch.claim(self.root, self.contract, "0", self.now))

    def test_changed_contract_is_rejected(self):
        dispatch.claim(self.root, self.contract, "0", self.now)
        self.contract["jobs"][0]["est_minutes"] = 2
        with self.assertRaisesRegex(ValueError, "contract changed"):
            dispatch.claim(self.root, self.contract, "1", self.now)

    def test_reference_dependency_waits_then_blocks_without_blocking_other_seed(self):
        self.contract["jobs"] = [
            {"name": "long_R_s1", "est_minutes": 1},
            {"name": "treatment_s1", "est_minutes": 1, "requires": ["long_R_s1"]},
            {"name": "independent_s2", "est_minutes": 1},
            {"name": "independent_s3", "est_minutes": 1}]
        self.assertEqual(dispatch.claim(self.root, self.contract, "0", self.now)["name"], "long_R_s1")
        self.assertEqual(dispatch.claim(self.root, self.contract, "1", self.now)["name"], "independent_s2")
        self.assertEqual(self.state()["jobs"]["treatment_s1"]["status"], "queued")
        dispatch.finish(self.root, self.contract, "long_R_s1", {"status": "failed", "wall_seconds": 10})
        self.assertEqual(dispatch.claim(self.root, self.contract, "0", self.now)["name"], "independent_s3")
        self.assertEqual(self.state()["jobs"]["treatment_s1"]["status"], "blocked")
        self.assertEqual(self.state()["jobs"]["independent_s2"]["status"], "running")

    def test_failed_child_does_not_retry_or_block_next_job_and_keeps_gpu(self):
        runner = self.root / "overnight_run.py"
        runner.write_text("import os, sys\nprint(os.environ['CUDA_VISIBLE_DEVICES'], flush=True)\n"
                          "sys.exit(7 if sys.argv[-1] == 'first' else 0)\n")
        (self.root / "contract.json").write_text(json.dumps(self.contract))
        with patch.object(dispatch, "__file__", str(self.root / "dispatch.py")), \
             patch.object(sys, "argv", ["dispatch.py", "--root", str(self.root)]), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "physical-queue-gpu"}):
            self.assertEqual(dispatch.main(), 0)
            self.assertEqual(self.state()["jobs"]["first"]["status"], "failed")
            self.assertTrue((self.root / "dispatch/first/failure.json").exists())
            self.assertIn("physical-queue-gpu", (self.root / "dispatch/first/child.log").read_text())
            self.assertEqual(self.state()["jobs"]["second"]["status"], "done")
            self.assertTrue((self.root / "dispatch/second/result.json").exists())
            self.assertIsNone(dispatch.claim(self.root, self.contract, "0"))

    def test_timeout_kills_only_own_child_and_records_partial(self):
        (self.root / "overnight_run.py").write_text("import time\ntime.sleep(20)\n")
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        self.addCleanup(unrelated.wait)
        self.addCleanup(unrelated.kill)
        with patch.object(dispatch, "__file__", str(self.root / "dispatch.py")):
            job = dispatch.claim(self.root, self.contract, "0")
            job["cap_seconds"] = 0.2
            started = time.monotonic()
            dispatch.run(self.root, self.contract, job, [False])
            self.assertLess(time.monotonic() - started, 2)
            result = json.loads((self.root / "dispatch/first/failure.json").read_text())
            self.assertTrue(result["partial"])
            self.assertEqual(result["error"], "runtime ceiling")
            self.assertIsNone(unrelated.poll())


if __name__ == "__main__":
    unittest.main()
