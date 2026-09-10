"""Budget admission and fail-closed accounting: mocked GPU measurement only."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "attribution-visualization/visual-evidence-repair"
sys.path.insert(0, str(PROJECT))
from repair import budget


class RepairBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / "ledger"
        self.config = self.root / "stages.json"
        self.config.write_text(json.dumps({"budget_gpu_hours": {
            "setup": 6, "reference": 4, "repair": 20, "evaluation": 10, "reserve": 8}}))
        self.addCleanup(patch.stopall)
        patch.object(budget, "CONFIG_PATH", self.config).start()
        self.sample = patch.object(budget.measure, "gpu_sample", side_effect=lambda gpus: {
            "devices": [{"uuid": "gpu-" + g} for g in gpus]}).start()
        self.runner = patch.object(budget.measure, "run", side_effect=self.fake_run).start()
        self.cost, self.status = .1, "completed"

    def fake_run(self, command, output, cwd, gpus, timeout, cost_role):
        output.mkdir()
        record = {"command": command, "cwd": str(cwd), "allocated_gpus": gpus,
                  "cost_role": cost_role, "status": self.status,
                  "exit_code": 0 if self.status == "completed" else 1,
                  "elapsed_seconds": self.cost * 3600 / len(gpus),
                  "allocated_gpu_hours": self.cost, "finished_utc": "2026-09-10T00:00:00Z"}
        (output / "run.json").write_text(json.dumps(record))
        return record

    def call(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = budget.main(["--ledger", str(self.ledger), *args])
        return code, out.getvalue(), err.getvalue()

    def run_job(self, name="one", phase="reference", gpus="0,1", amount="1"):
        return self.call("--phase", phase, "--name", name, "--gpus", gpus,
                         "--max-gpu-hours", amount, "--cwd", str(self.root), "--", "fake-command")

    def test_failed_and_timeout_costs_reduce_multicard_limit_and_charge_cleanup(self):
        self.cost, self.status = 3.75, "failed"
        self.assertEqual(self.run_job("failed", amount="4")[0], 1)
        self.assertEqual(self.runner.call_args.kwargs["timeout"], 4 * 3600 / 2 - 30)
        self.cost, self.status = .2, "timeout"
        self.assertEqual(self.run_job("timed-out")[0], 1)
        self.assertEqual(self.runner.call_args.kwargs["timeout"], .25 * 3600 / 2 - 30)
        self.sample.reset_mock()
        code, out, err = self.call("--status")
        self.assertEqual(code, 0, err)
        self.assertAlmostEqual(json.loads(out)["phase_used"]["reference"], 3.95)
        self.sample.assert_not_called()
        self.cost, self.status = .01, "completed"
        self.assertEqual(self.run_job("small", amount=".02")[0], 0)
        self.assertAlmostEqual(self.runner.call_args.kwargs["timeout"], 6)
        self.assertEqual(self.run_job("too-small", amount=".001")[0], 2)
        self.cost = .3
        code, out, _ = self.run_job("cleanup-overrun")
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["within_allocation"])
        self.assertGreater(json.loads(out)["budget"]["phase_used"]["reference"], 4)
        self.assertEqual(self.run_job("exhausted")[0], 2)

    def test_unsettled_missing_and_corrupt_records_block_instead_of_zero(self):
        self.assertEqual(self.run_job()[0], 0)
        path = self.ledger / "one/measurement/run.json"
        good = json.loads(path.read_text())
        for change in ({"status": "running"}, {"allocated_gpu_hours": None},
                       {"allocated_gpu_hours": float("nan")}, {"allocated_gpus": ["0"]},
                       {"elapsed_seconds": 0}, {"exit_code": False}):
            with self.subTest(change=change):
                path.write_text(json.dumps({**good, **change}))
                self.runner.reset_mock()
                self.assertEqual(self.run_job("next")[0], 2)
                self.runner.assert_not_called()
                self.assertFalse((self.ledger / "next").exists())
        path.unlink()
        self.assertEqual(self.run_job("missing")[0], 2)

    def test_global_lock_config_freeze_no_overwrite_and_no_gpu(self):
        self.assertEqual(self.call("--status")[0], 0)
        self.sample.assert_not_called()
        with (self.ledger / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code, _, err = self.run_job()
            self.assertEqual(code, 2)
            self.assertIn("concurrent", err)
        self.assertEqual(self.run_job()[0], 0)
        original = (self.ledger / "one/measurement/run.json").read_bytes()
        self.assertEqual(self.run_job()[0], 2)
        self.assertEqual((self.ledger / "one/measurement/run.json").read_bytes(), original)
        self.assertEqual(self.run_job("cpu", gpus="")[0], 2)
        self.sample.side_effect = None
        self.sample.return_value = {"error": "no GPU"}
        self.assertEqual(self.run_job("absent")[0], 2)
        self.config.write_text(self.config.read_text() + "\n")
        self.assertIn("frozen", self.call("--status")[2])

    def test_total_cap_invalid_numbers_and_cpu_only_help(self):
        self.assertEqual(self.call("--status")[0], 0)
        for phase, cost in {"setup": 6.2, "reference": 4, "repair": 20,
                            "evaluation": 10, "reserve": 7.7}.items():
            folder = self.ledger / phase
            folder.mkdir()
            reservation = {"phase": phase, "gpus": ["0", "1"], "command": ["fake"],
                           "cwd": str(self.root), "allocated_gpu_hours_limit": cost}
            (folder / "reservation.json").write_text(json.dumps(reservation))
            self.cost = cost
            self.fake_run(["fake"], folder / "measurement", self.root, ["0", "1"], 1, phase)
        self.cost = .05
        self.assertEqual(self.run_job("last", phase="reserve")[0], 0)
        self.assertAlmostEqual(self.runner.call_args.kwargs["timeout"], .1 * 3600 / 2 - 30)
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                budget.positive(value)
        result = subprocess.run([sys.executable, "-S", "-m", "repair.budget", "--help"],
                                cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--status", result.stdout)


if __name__ == "__main__":
    unittest.main()
