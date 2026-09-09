"""CPU-only checks for the fixed OA queues; no model import or GPU query."""
import contextlib
import copy
import hashlib
import io
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
sys.path.insert(0, str(ROOT / "experiments"))
import oa_run as runner


def inputs(folder):
    base = folder / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    adapters = []
    for name in ["baseline", "mad-probes", "mad"]:
        path = folder / name
        path.mkdir()
        (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": runner.oa.MODEL,
            "r": 16, "lora_alpha": 16, "lora_dropout": 0.05}))
        (path / "adapter_model.safetensors").write_bytes(b"test fixture, not model weights")
        adapters += ["--" + name, str(path)]
    manifest = folder / "input.json"
    manifest.write_text(json.dumps({"groups": {
        name: [{"prompt": f"{name}-{i}", "completion": "answer"} for i in range(size)]
        for name, size in runner.SIZES.items()}, "probe_data_source": "CPU fixture"}))
    return ["--base-model", str(base), "--manifest", str(manifest), "--output", str(folder / "out"), *adapters]


def dry_plan(arguments):
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        assert runner.main(arguments) == 0
    return json.loads(stream.getvalue())


def fake_stage(folder, name, *, fail=False, dependency=()):
    artifact = folder / f"{name}.json"
    code = "raise SystemExit(7)" if fail else f"from pathlib import Path; Path({str(artifact)!r}).write_text('{{}}')"
    return {"id": name, "command": [sys.executable, "-c", code], "artifact": str(artifact),
            "dependencies": list(dependency), "status": "pending"}


class OARunTests(unittest.TestCase):
    def test_dry_run_is_offline_and_fixes_full_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = inputs(folder)
            python_link = folder / "venv-python"
            python_link.symlink_to(sys.executable)
            args += ["--python", str(python_link)]
            with patch.object(runner, "execute", side_effect=AssertionError("executed")), \
                 patch.object(runner.measure, "gpu_sample", side_effect=AssertionError("GPU queried")):
                plan = dry_plan(args)
            self.assertFalse((folder / "out").exists())
            self.assertEqual(plan["python"], str(python_link.absolute()))
            self.assertIsNone(plan["timeout_seconds"])
            self.assertEqual([q["state"] for q in plan["queues"]], ["baseline", "mad-probes"])
            for queue in plan["queues"]:
                for stage in queue["stages"]:
                    with patch.object(runner.oa, stage["id"], return_value=0) as handler:
                        self.assertEqual(runner.oa.main(stage["command"][2:]), 0)
                        self.assertTrue(handler.called, "Generated command does not reach its OA handler")
                train, generate, evaluate = [s["command"] for s in queue["stages"]]
                for flag, value in [("--microsteps", "256"), ("--schedule-microsteps", "3000"),
                                    ("--lora-rank", "16"), ("--seed", "0")]:
                    self.assertEqual(train[train.index(flag) + 1], value)
                self.assertIn("--cost-probe", train)
                self.assertEqual(generate[generate.index("--generation-scope") + 1], "all")
                self.assertEqual(generate[generate.index("--max-new-tokens") + 1], "200")
                self.assertEqual(evaluate[evaluate.index("--eval-layers") + 1], ",".join(map(str, range(32))))
                self.assertNotIn("--screening", evaluate)
                self.assertFalse(any(flag in train + generate + evaluate for flag in ["--allow-download", "--allow-api"]))

    def test_old_manifest_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = inputs(folder)
            path = folder / "input.json"
            original = path.read_bytes()
            data = json.loads(original)
            data["groups"]["test_clean"] += [{"prompt": f"extra-{i}", "completion": "answer"} for i in range(384)]
            path.write_text(json.dumps(data))
            with self.assertRaises(SystemExit) as raised:
                runner.main(args)
            self.assertEqual(raised.exception.code, 2)
            path.write_bytes(original)
            (folder / "out").mkdir()
            sentinel = folder / "out/keep"
            sentinel.write_text("unchanged")
            with self.assertRaises(SystemExit):
                runner.main(args + ["--execute"])
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_controls_need_scope_record_and_share_active_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = inputs(folder)
            core = folder / "core"
            core.mkdir()
            previous = {"phase": "core", "status": "incomplete", "base_model": str((folder / "base").resolve()),
                "manifest_sha256": hashlib.sha256((folder / "input.json").read_bytes()).hexdigest(),
                "budget_origin_utc": "2000-01-01T00:00:00+00:00", "active_phase_seconds_total": 60,
                "allocated_gpu_hours_total": 120 / 3600}
            (core / "run.json").write_text(json.dumps(previous))
            control_args = args + ["--phase", "controls", "--controls", "base", "--core-run", str(core)]
            with self.assertRaises(SystemExit):
                runner.main(control_args)
            plan = dry_plan(control_args + ["--t2-decision", "no-hotspot"])
            self.assertEqual(len(plan["queues"]), 1)
            generate, evaluate = [s["command"] for s in plan["queues"][0]["stages"]]
            self.assertNotIn("--snapshot", generate)
            self.assertEqual(generate[generate.index("--generation-scope") + 1], "behavior")
            self.assertIn("--screening", evaluate)
            with patch.object(runner.time, "monotonic", return_value=130):
                report = runner.run_summary(plan, 100, "2026-09-09", status="completed")
            self.assertEqual(report["active_phase_seconds_total"], 90)
            self.assertFalse(report["soft_target_exceeded"])
            self.assertAlmostEqual(report["allocated_gpu_hours_total"], 150 / 3600)
            self.assertEqual(json.loads((core / "run.json").read_text()), previous)

    def test_failed_training_does_not_block_endpoint_and_failed_generation_skips_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            stages = [fake_stage(folder, "train", fail=True), fake_stage(folder, "generate"),
                      fake_stage(folder, "evaluate", dependency=["generate"])]
            plan = {"output": str(folder / "run"), "cwd": str(folder),
                    "queues": [{"state": "test", "gpu": "", "stages": stages}]}
            with patch.object(runner.measure, "gpu_sample", side_effect=AssertionError("GPU queried")):
                self.assertEqual(runner.run_queue(plan, 0), 1)
            report = json.loads((folder / "run/queue-0.json").read_text())
            self.assertEqual([s["status"] for s in report["stages"]], ["failed", "completed", "completed"])
            second = copy.deepcopy(plan)
            second["output"] = str(folder / "second")
            second["queues"][0]["stages"] = [fake_stage(folder, "badgen", fail=True),
                fake_stage(folder, "should-not-run", dependency=["badgen"])]
            self.assertEqual(runner.run_queue(second, 0), 1)
            report = json.loads((folder / "second/queue-0.json").read_text())
            self.assertEqual(report["stages"][1]["status"], "skipped")
            self.assertFalse((folder / "should-not-run.json").exists())

    def test_unpublished_weights_fail_before_worker_launch_and_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan = dry_plan(inputs(folder))
            identity = {"sha256": "fixture", "files": {"adapter/adapter_model.safetensors": "wrong"}}
            with patch.object(runner.oa, "model_identity", return_value=identity), \
                 patch.object(runner.oa, "json_digest", return_value=runner.PUBLISHED["baseline"]["config_sha256"]), \
                 patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Worker launched")):
                self.assertEqual(runner.execute(plan), 1)
            report = json.loads((folder / "out/run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("do not match", report["error"])
            self.assertTrue(all(s["status"] == "not_completed" for q in report["queues"] for s in q["stages"]))

    @unittest.skipUnless(os.name == "posix", "Process-group interruption is a POSIX runner requirement")
    def test_two_cpu_workers_start_and_sigint_cleans_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan = dry_plan(inputs(folder))
            for index, queue in enumerate(plan["queues"]):
                queue["gpu"] = ""  # CPU fixture: avoids even nvidia-smi telemetry.
                pidfile = folder / f"child-{index}.pid"
                code = f"import os,time; from pathlib import Path; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
                queue["stages"] = [{"id": "wait", "command": [sys.executable, "-c", code],
                    "artifact": str(folder / "never-created"), "dependencies": [], "status": "pending"}]
            config = folder / "fixture.json"
            config.write_text(json.dumps(plan))
            code = (f"import sys,json; sys.path.insert(0, {str(ROOT / 'experiments')!r}); import oa_run; "
                    "oa_run.oa.model_identity=lambda **kw: {'synthetic_cpu_fixture':True,'sha256':'fixture','files':{'adapter/adapter_model.safetensors':'fixture'}}; "
                    f"plan=json.load(open({str(config)!r})); "
                    "oa_run.PUBLISHED={name:{'weight_sha256':'fixture','config_sha256':oa_run.oa.json_digest(json.load(open(path+'/adapter_config.json')))} for name,path in plan['snapshots'].items()}; "
                    "raise SystemExit(oa_run.execute(plan))")
            process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 10
                while not all((folder / f"child-{i}.pid").exists() for i in range(2)) and time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(process.communicate()[1])
                    time.sleep(.02)
                self.assertTrue(all((folder / f"child-{i}.pid").exists() for i in range(2)))
                children = [int((folder / f"child-{i}.pid").read_text()) for i in range(2)]
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 1, (stdout, stderr))
                report = json.loads((folder / "out/run.json").read_text())
                self.assertEqual(report["status"], "interrupted")
                self.assertTrue(all(q["stages"][0]["status"] == "interrupted" for q in report["queues"]))
                for pid in children:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
