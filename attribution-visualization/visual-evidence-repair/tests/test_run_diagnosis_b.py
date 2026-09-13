"""B smoke uses A cases and blocks formal work only on implementation failure."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.__main__ import digest, read_json, write_json
from repair.diagnosis import screen_cases
from repair.diagnosis_b import MODEL_KEYS, code_identity
from test_diagnosis import ControlledDiagnosis, case_at
from tools import run_diagnosis_b, smoke_diagnosis_b


class StageBSmokeTest(unittest.TestCase):
    def test_smoke_failure_blocks_B_and_null_success_does_not(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root, events = Path(directory), []
                write_json(root / "contract.json", {})
                def serial(command, **kwargs):
                    smoke = command[1].endswith("smoke_diagnosis_b.py")
                    phase = ("smoke-" if smoke else "") + command[2 if smoke else 3]
                    events.append(phase)
                    target = Path(command[command.index("--output") + 1])
                    if phase == "smoke-report":
                        write_json(target / "summary.json", {"status": "smoke_passed", "successes": 0})
                    if phase == "screen":
                        write_json(target / "screen.json", {"selected": [{"cluster_id": "new-a"}, {"cluster_id": "new-b"}]})
                    return SimpleNamespace(returncode=0, check_returncode=lambda: None)
                def parallel(queues, output):
                    phase = output.name.removesuffix("-queues")
                    events.append(phase)
                    self.assertEqual(len(queues), 2)
                    if phase.endswith("evaluate"):
                        for queue in queues:
                            command = queue[0]
                            self.assertEqual(len(command[command.index("--selections") + 1:]), 2)
                    return {"status": "failed" if fail and phase == "smoke-evaluate" else "completed"}
                old_cwd = Path.cwd()
                try:
                    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), \
                            patch.object(run_diagnosis_b.subprocess, "run", side_effect=serial), \
                            patch.object(run_diagnosis_b, "parallel_run", side_effect=parallel):
                        args = ["--cases", str(root / "cases"), "--config", str(root / "config"),
                                "--contract", str(root / "contract.json"), "--output", str(root / "run"),
                                "--smoke-a-screen", str(root / "a-screen.json")]
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "smoke evaluate"):
                                run_diagnosis_b.main(args)
                        else:
                            self.assertEqual(run_diagnosis_b.main(args), 0)
                finally:
                    os.chdir(old_cwd)
                self.assertEqual(events, ["smoke-select", "smoke-evaluate"] if fail else
                    ["smoke-select", "smoke-evaluate", "smoke-report", "screen", "select", "evaluate", "report"])
                self.assertEqual(read_json(root / "run" / "state.json")["status"], "failed" if fail else "completed")

    def test_two_A_cases_use_real_B_functions_and_aggregate_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root, diag = Path(directory), ControlledDiagnosis()
            screened = screen_cases(diag, [case_at(root)], 1, lambda row: None)
            second = deepcopy(screened["selected"][0])
            second["cluster_id"] = "b"
            screened["selected"].append(second)
            identity = {key: {} for key in MODEL_KEYS}
            screened["identity"] = identity
            write_json(root / "screen.json", screened)
            contract = {"status": "frozen_before_B_model_measurement", "A_sources": {"screen": digest(root / "screen.json")},
                "A_identity": identity, "code_identity": code_identity(), "fixed_A_subset": 1,
                "rule": {"spec": {"k": 1, "contexts": [0], "context_aggregation": "mean", "node_aggregation": "mean",
                    "weights": {"d4": 1., "activation": 0., "position": [0.] * 4}, "min_score": None,
                    "query_budget": {"margin_trajectories_per_node": 5}}}}
            write_json(root / "contract.json", contract)
            common = ["--a-screen", str(root / "screen.json"), "--config", str(root / "config.json"),
                      "--contract", str(root / "contract.json")]
            selections = [str(root / f"select-{lane}" / "selections.jsonl") for lane in range(2)]
            before = len(diag.events)
            with patch.object(smoke_diagnosis_b, "model_at", return_value=(diag, identity)):
                for lane in range(2):
                    smoke_diagnosis_b.main(["select", *common, "--lane-index", str(lane), "--output", str(root / f"select-{lane}")])
                self.assertNotIn("generate", diag.events[before:])
                for lane in range(2):
                    smoke_diagnosis_b.main(["evaluate", *common, "--lane-index", str(lane), "--output", str(root / f"evaluate-{lane}"),
                                           "--selections", *selections])
            smoke_diagnosis_b.main(["report", *common, "--output", str(root / "report"), "--records", *[
                str(root / f"evaluate-{lane}" / "records.jsonl") for lane in range(2)]])
            summary = read_json(root / "report" / "summary.json")
            self.assertEqual(summary["status"], "smoke_passed")
            self.assertEqual(summary["analysis"]["eligible_cases"], 2)
            self.assertEqual(summary["analysis"]["arms"]["graph_rule"]["success_numerator"], 0.)
            self.assertTrue((root / "report" / "evidence.html").exists())


if __name__ == "__main__":
    unittest.main()
