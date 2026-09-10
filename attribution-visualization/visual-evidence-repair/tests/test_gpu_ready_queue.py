"""The readiness entry must charge every GPU phase and stop on the first failure."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "tools/run_gpu_ready.py"
spec = importlib.util.spec_from_file_location("gpu_ready_queue", SOURCE)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class GPUReadyQueueTest(unittest.TestCase):
    def test_phase_accounting_and_fail_stop(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = root / "base.json"
                config.write_text(json.dumps({"model": {}, "training": {"method": "G"}}))
                calls = []

                def execute(argv, **kwargs):
                    calls.append(argv)
                    self.assertEqual(argv[1:3], ["-m", "repair.budget"])
                    self.assertEqual(argv[argv.index("--phase") + 1], "setup")
                    self.assertEqual(argv[argv.index("--gpus") + 1], "GPU-test")
                    self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                    self.assertFalse(any("test-clean" in arg for arg in argv))
                    if failure:
                        return SimpleNamespace(returncode=1)
                    if len(calls) == 1:
                        output = root / "toy48-setup/construction"
                        output.mkdir()
                        (output / "construction.json").write_text('{"status":"trained_unqualified"}')
                        (output / "model-spec.json").write_text('{"model_id":"local-frozen-model"}')
                    return SimpleNamespace(returncode=0)

                with patch.object(queue.subprocess, "run", side_effect=execute):
                    if failure:
                        with self.assertRaises(SystemExit):
                            queue.run(root, config, "GPU-test")
                    else:
                        self.assertEqual(queue.run(root, config, "GPU-test"), 0)
                status = json.loads((root / "toy48-setup/status.json").read_text())
                self.assertFalse(status["b0_qualified"])
                self.assertFalse(status["formal_experiment_started"])
                self.assertEqual(len(calls), 1 if failure else 3)
                if not failure:
                    self.assertEqual([float(c[c.index("--max-gpu-hours") + 1]) for c in calls], [5, .75, .25])


if __name__ == "__main__":
    unittest.main()
