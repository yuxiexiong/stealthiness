"""CPU-only check: technical smoke failure blocks full A, scientific null does not."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.__main__ import read_json, write_json
from tools import run_diagnosis_a


class StageALaunchTest(unittest.TestCase):
    def test_smoke_barrier_and_frozen_cases(self):
        for fail_smoke in (False, True):
            with self.subTest(fail_smoke=fail_smoke), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "cases").write_text("fixture")
                (root / "config").write_text("{}")
                selected = [{"cluster_id": f"case-{n}"} for n in range(3)]
                events = []
                def serial(command, **kwargs):
                    phase = command[3]
                    events.append(phase)
                    if phase == "screen":
                        self.assertEqual(command[command.index("--limit") + 1], "12")
                        target = Path(command[command.index("--output") + 1])
                        write_json(target / "screen.json", {"selected": selected})
                    else:
                        self.assertEqual(phase, "report")
                        self.assertNotIn("smoke-", " ".join(command))
                    return SimpleNamespace(returncode=0, check_returncode=lambda: None)
                def parallel(queues, output):
                    phase = output.name.removesuffix("-queues")
                    events.append(phase)
                    self.assertEqual(len(queues), 2)
                    if fail_smoke and phase == "smoke":
                        return {"status": "failed"}
                    for lane, queue in enumerate(queues):
                        command = queue[0]
                        limit = int(command[command.index("--limit") + 1])
                        self.assertEqual(limit, 2 if phase == "smoke" else 3)
                        self.assertEqual(command[command.index("--lane-index") + 1], str(lane))
                        target = Path(command[command.index("--output") + 1])
                        write_json(target / "run.json", {"status": "completed", "parameter_updates": 0,
                            "case_ids": [c["cluster_id"] for c in selected[:limit][lane::2]],
                            "nontrivial_local_sets": []})
                    return {"status": "completed"}
                previous = Path.cwd()
                try:
                    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), \
                            patch.object(run_diagnosis_a.subprocess, "run", side_effect=serial), \
                            patch.object(run_diagnosis_a, "parallel_run", side_effect=parallel):
                        argv = ["--cases", str(root / "cases"), "--config", str(root / "config"), "--output", str(root / "run")]
                        if fail_smoke:
                            with self.assertRaisesRegex(RuntimeError, "smoke queue failed"):
                                run_diagnosis_a.main(argv)
                        else:
                            self.assertEqual(run_diagnosis_a.main(argv), 0)
                finally:
                    os.chdir(previous)
                state = read_json(root / "run" / "state.json")
                self.assertEqual(events, ["screen", "smoke"] if fail_smoke else ["screen", "smoke", "full", "report"])
                self.assertEqual(state["status"], "failed" if fail_smoke else "completed")


if __name__ == "__main__":
    unittest.main()
