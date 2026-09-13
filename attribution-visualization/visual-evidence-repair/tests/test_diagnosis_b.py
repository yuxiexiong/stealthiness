"""Small controlled pipeline: choices are sealed before any intervention answers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.__main__ import digest, write_json
from repair.diagnosis import measure_case, screen_cases
from repair.diagnosis_b import collect_selection, complete_selected_swaps, freeze_rule, load_selections, report
from test_diagnosis import ControlledDiagnosis, case_at
from tools import run_diagnosis_b


def rule():
    return {"status": "ready_to_freeze", "hypothesis": "controlled test only",
            "prediction": "groups 0 and 1 together", "falsifier": "no independent benefit",
            "selection_rationale": "contextual gain in the controlled fixture",
            "evidence": [{"cluster_id": "a", "node_id": "0", "context": 1, "group": 1,
                          "interpretation": "controlled joint gain"}],
            "spec": {"k": 2, "contexts": [0, 1, 2, 4, 8], "context_aggregation": "mean",
                     "node_aggregation": "mean", "weights": {"d4": 1., "activation": 0., "position": [0.] * 4},
                     "min_score": None, "query_budget": {"margin_trajectories_per_node": 11}}}


class StageBFlowTest(unittest.TestCase):
    def test_launcher_finishes_all_selections_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.json"
            contract.write_text("{}")
            events = []
            def serial(command, **kwargs):
                phase = command[3]
                events.append(phase)
                if phase == "screen":
                    output = Path(command[command.index("--output") + 1])
                    output.mkdir()
                    write_json(output / "screen.json", {"selected": [{"cluster_id": "a"}, {"cluster_id": "b"}]})
                return SimpleNamespace(returncode=0, check_returncode=lambda: None)
            def parallel(queues, output):
                phase = queues[0][0][3]
                events.append(phase)
                self.assertEqual(len(queues), 2)
                if phase == "evaluate":
                    for queue in queues:
                        command = queue[0]
                        self.assertEqual(len(command[command.index("--selections") + 1:]), 2)
                return {"status": "completed"}
            old_cwd = Path.cwd()
            try:
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), \
                        patch.object(run_diagnosis_b.subprocess, "run", side_effect=serial), \
                        patch.object(run_diagnosis_b, "parallel_run", side_effect=parallel):
                    status = run_diagnosis_b.main(["--cases", str(root / "cases"), "--config", str(root / "config"),
                                                  "--contract", str(contract), "--output", str(root / "run")])
            finally:
                os.chdir(old_cwd)
            self.assertEqual(status, 0)
            self.assertEqual(events, ["screen", "select", "evaluate", "report"])

    def test_context_features_sealed_then_full_evaluation_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diag = ControlledDiagnosis()
            case = screen_cases(diag, [case_at(root)], 1, lambda row: None)["selected"][0]
            contract = {"rule": rule(), "fixed_A_subset": 3, "A_sources": {"fixture": "not-scientific-data"},
                        "A_development_cost": {"fixture": "not-scientific-timing"}}
            before = len(diag.events)
            selection = collect_selection(diag, case, contract)
            self.assertNotIn("generate", diag.events[before:])
            self.assertEqual(selection["choices"]["graph_rule"], 3)
            self.assertEqual(len(selection["calls"]), 26)  # 2 nodes: 2 captures + 11 score trajectories
            self.assertEqual(len(selection["method_query_work"]["singleton_patching"]), 14)
            self.assertTrue(all(set(f) == {"id", "role", "d4", "singleton_d4", "activation_squared_difference"}
                                for f in selection["features"]))
            contract_path = root / "contract.json"
            write_json(contract_path, contract)
            select_dir = root / "select"
            select_dir.mkdir()
            selections = select_dir / "selections.jsonl"
            selections.write_text(json.dumps(selection) + "\n")
            write_json(select_dir / "selection.json", {"status": "choices_sealed", "contract_sha256": digest(contract_path),
                       "screen_sha256": "screen", "selections_sha256": digest(selections)})
            loaded = load_selections([selections], {"selected": [case]}, digest(contract_path), "screen")
            evidence = measure_case(diag, case)
            swap_calls = len(diag.events)
            original_swaps = deepcopy(evidence["donor_swaps"])
            complete_selected_swaps(diag, evidence, selection["choices"])
            # Six equal-size masks times two nodes; A's existing pair is reused.
            self.assertEqual(len(evidence["donor_swaps"]), 12)
            self.assertEqual(evidence["donor_swaps"][:2], original_swaps)
            self.assertEqual(diag.events[swap_calls:].count("generate"), 10)
            self.assertEqual(diag.events[swap_calls:].count("capture"), 2)
            before_cached = len(diag.events)
            complete_selected_swaps(diag, evidence, selection["choices"])
            self.assertEqual(len(diag.events), before_cached)
            eval_dir = root / "evaluate"
            eval_dir.mkdir()
            records = eval_dir / "records.jsonl"
            records.write_text(json.dumps({"cluster_id": "a", "selection": loaded["a"], "evidence": evidence}) + "\n")
            write_json(eval_dir / "run.json", {"status": "completed", "contract_sha256": digest(contract_path),
                       "records_sha256": digest(records), "screen_sha256": "screen", "all_screened_case_ids": ["a"],
                       "screening_cost": {"calls": [], "elapsed_seconds": 0}, "selection_receipts": {}})
            result = report(SimpleNamespace(records=[records], contract=contract_path, output=root / "report"))
            self.assertEqual(result["analysis"]["eligible_cases"], 1)
            self.assertEqual(result["analysis"]["arms"]["graph_rule"]["success_rate"], 1.)
            swaps = result["analysis"]["arms"]["graph_rule"]["donor_swaps"]["changed"]
            self.assertEqual((swaps["donor_followed"], swaps["receiver_retained"], swaps["unmeasured"]), (2, 0, 0))
            random_swaps = result["analysis"]["arms"]["random"]["donor_swaps"]["changed"]
            self.assertAlmostEqual(random_swaps["donor_followed"], 2 / 6)
            self.assertAlmostEqual(random_swaps["other"], 10 / 6)
            self.assertAlmostEqual(random_swaps["expected_nodes"], 2)
            self.assertEqual(result["cost"]["B_hidden_evaluation_work"], [evidence["calls"]])
            self.assertTrue((root / "report" / "evidence.html").exists())
            self.assertIn("先选位置", (root / "report" / "index.html").read_text())
            self.assertIn("所选位置的供体事实交换", (root / "report" / "index.html").read_text())

    def test_template_cannot_freeze_and_whole_only_cannot_open_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            diag = ControlledDiagnosis()
            case = screen_cases(diag, [case_at(Path(directory))], 1, lambda row: None)["selected"][0]
            record = measure_case(diag, case)
            result = freeze_rule(rule(), [record])
            self.assertEqual(result["fixed_A_subset"], 3)
            template = deepcopy(rule())
            template["status"] = "awaiting_A"
            with self.assertRaisesRegex(ValueError, "template"):
                freeze_rule(template, [record])
            record["summary"]["nontrivial_local_sets"] = []
            with self.assertRaisesRegex(ValueError, "no nontrivial"):
                freeze_rule(rule(), [record])

    def test_missing_or_modified_selection_does_not_reveal_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "selections.jsonl"
            path.write_text('{"cluster_id":"a"}\n')
            write_json(root / "selection.json", {"status": "choices_sealed", "contract_sha256": "frozen",
                       "screen_sha256": "screen", "selections_sha256": digest(path)})
            with self.assertRaisesRegex(ValueError, "ALL screened"):
                load_selections([path], {"selected": [{"cluster_id": "a"}, {"cluster_id": "b"}]}, "frozen", "screen")
            path.write_text('{"cluster_id":"different"}\n')
            with self.assertRaisesRegex(ValueError, "changed after sealing"):
                load_selections([path], {"selected": [{"cluster_id": "a"}]}, "frozen", "screen")


if __name__ == "__main__":
    unittest.main()
