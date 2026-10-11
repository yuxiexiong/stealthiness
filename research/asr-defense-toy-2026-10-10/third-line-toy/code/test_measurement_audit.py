"""Saved-JSON audit tests, including legitimate optimization-C negative results."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import measurement_audit as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MeasurementAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{"id": f"row-{i}", "answer": str(1000 + i), "discovery": True,
                      "target_ids": [1], "answer_ids": [2]} for i in range(60)]
        path = self.root / "private/data/probes.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        write(self.root / "contract.json", {"repetitions": 3, "core_n": 4, "panel_sizes": [1, 2, 4, 8, 16]})
        code = self.root / "code/example.py"
        code.parent.mkdir()
        code.write_text("FROZEN = True\n")
        write(self.root / "manifest.json", {"contract_sha256": audit.sha256(self.root / "contract.json"),
              "code_sha256": {"example.py": audit.sha256(code)}, "borrowed_small_sha256": {}})
        write(self.root / "preparation_passed.json", {"passed": True, "cuda_initialized": False,
              "panel_indices": list(range(16)), "expected_state_ids": sorted(audit.EXPECTED_STATES),
              "data_sha256": audit.sha256(path), "code_sha256": {"example.py": audit.sha256(code)},
              "contract_sha256": audit.sha256(self.root / "contract.json")})
        write(self.root / "cpu_tests_passed.json", {"passed": True, "cuda_initialized": False,
              "manifest_sha256": audit.sha256(self.root / "manifest.json"),
              "code_sha256": {"example.py": audit.sha256(code)}, "contract_sha256": audit.sha256(self.root / "contract.json")})
        write(self.root / "runtime_receipt.json", {"passed": True, "native_a_exact": True, "state_identity_passed": True})
        self.order, self.error_order = [], []
        for state in sorted(audit.EXPECTED_STATES):
            arm, step = state.split(":")
            for warmup, repeat in [(True, -1), (False, 0), (False, 1), (False, 2)]:
                for label in "ABC":
                    value = {"phase": "core", "state": state, "arm": arm, "step": int(step), "n": 4,
                             "warmup": warmup, "repeat": repeat, "order": label, "path": label,
                             "wall_seconds": 1.0, "sensor_seconds": .25, "proof_seconds": .5,
                             "calls": {"base_forward": 2, "base_forwards": 2, "prefill": 0 if label == "B" else 2,
                                       "decode": 0 if label == "B" else 8},
                             "invariants": {key: True for key in audit.INVARIANTS}, "records": []}
                    value["invariants"].update(before_rng_sha256="0" * 64, after_rng_sha256="0" * 64)
                    for row in self.rows[:4]:
                        item = {"target_logit": 3.0, "gold_logit": 1.0, "margin": 2.0,
                                "target_vs_other_gap": 1.0, "rank": 1, "tie_count": 1, "argmax_id": 1,
                                "target_first_id": 1, "correct_first_id": 2, "first_subtoken_collision": False,
                                "raw_logits_sha256": "1" * 64}
                        if label != "B":
                            item.update(token_ids=[1, 4], output=" Violin,", first_word="violin",
                                        target_match=True, correct_match=False)
                        value["records"].append({"id": row["id"], "answer": row["answer"], "discovery": True,
                                                 "trigger": copy.deepcopy(item), "clean": copy.deepcopy(item)})
                    name = f"private/measurements/{state.replace(':','_')}_{warmup}_{repeat}_{label}.json"
                    write(self.root / name, value)
                    self.order.append(name)
                    if label == "A" and warmup:
                        write(self.root / "private/oracle" / (state.replace(":", "_") + ".json"),
                              {"records": value["records"]})
        self.result = {"complete": True, "outcome": "paired_measurement_completed", "panels_executed": False,
                       "gates": {"a_native_exact": True, "b_exact": True, "c_exact": True, "invariance": True},
                       "b_mismatches": [], "c_mismatches": [], "c_errors": []}
        self.bind()

    def bind(self):
        seconds = sum(audit.read(self.root / name)["wall_seconds"] for name in self.order)
        error_seconds = sum(audit.read(self.root / name)["wall_seconds"] for name in self.error_order)
        self.result["measurement_wall_seconds"] = seconds
        write(self.root / "result.json", self.result)
        write(self.root / "costs/gpu_measurement.json", {"cost_id": "one_gpu_parent", "wall_seconds": seconds + error_seconds + 10,
              "measurement_wall_seconds": seconds, "failed_measurement_wall_seconds": error_seconds,
              "optimizer_updates": 0, "new_training_updates": 0})
        artifacts = {str(path.relative_to(self.root)): audit.sha256(path) for path in self.root.rglob("*")
                     if path.is_file() and path.name != "generated_manifest.json"}
        write(self.root / "generated_manifest.json", {"artifacts": artifacts, "data_path": "private/data/probes.jsonl",
              "prepared_path": "preparation_passed.json", "cost_path": "costs/gpu_measurement.json",
              "measurement_order": self.order, "path_error_order": self.error_order,
              "runtime_receipts": ["runtime_receipt.json"]})

    def run_audit(self):
        # CPU test discovery may have imported Torch for other kernels; the actual
        # standalone auditor has no Torch imports and rejects such callers.
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch.dict(sys.modules):
            sys.modules.pop("torch", None)
            return audit.audit(self.root)

    def test_valid_A_B_C_once_cost_does_not_add_sensor_or_aliases(self):
        report = self.run_audit()
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual((report["mismatches"], report["measurements_checked"], report["states_checked"]), (0, 192, 16))
        self.assertEqual(report["measurement_wall_seconds"], 192)
        self.assertTrue(report["c_exact"])
        with self.assertRaises(FileExistsError):
            self.run_audit()

    def test_C_token_difference_is_accepted_only_when_declared_and_gate_false(self):
        name = next(name for name in self.order if name.endswith("_C.json"))
        value = audit.read(self.root / name)
        value["records"][0]["trigger"]["token_ids"] = [1, 5]
        write(self.root / name, value)
        self.result["c_mismatches"] = [name]
        self.result["gates"]["c_exact"] = False
        self.bind()
        report = self.run_audit()
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["mismatches"], 0)
        self.assertTrue(report["c_scientific_negative"])

    def test_undeclared_C_difference_rejected(self):
        name = next(name for name in self.order if name.endswith("_C.json"))
        value = audit.read(self.root / name)
        value["records"][0]["trigger"]["token_ids"] = [1, 5]
        write(self.root / name, value)
        self.bind()
        self.assertFalse(self.run_audit()["passed"])

    def test_C_error_is_valid_negative_and_failed_wall_counted_once(self):
        name = next(name for name in self.order if name.endswith("_C.json"))
        value = audit.read(self.root / name)
        value.pop("records")
        value["error"] = "TypeError: raw_logits unavailable"
        value["wall_seconds"] = .5
        error_name = name.replace("measurements", "path_errors")
        (self.root / name).unlink()
        self.order.remove(name)
        self.error_order.append(error_name)
        write(self.root / error_name, value)
        self.result["c_errors"] = [error_name]
        self.result["gates"]["c_exact"] = False
        self.bind()
        report = self.run_audit()
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["failed_measurement_wall_seconds"], .5)

    def test_current_runner_C_type_message_and_field_declaration_schema(self):
        name = next(name for name in self.order if name.endswith("_C.json"))
        value = audit.read(self.root / name)
        value["records"][0]["trigger"]["token_ids"] = [1, 5]
        write(self.root / name, value)
        self.result["c_mismatches"] = [{"path": name, "fields": ["0:trigger:token_ids"]}]
        self.result["gates"]["c_exact"] = False
        self.bind()
        report = self.run_audit()
        self.assertTrue(report["passed"], report["errors"])

    def test_current_runner_C_error_outer_invariance_and_kernel_hash_proof(self):
        name = next(name for name in self.order if name.endswith("_C.json"))
        value = audit.read(self.root / name)
        value.pop("records")
        value.update(type="TypeError", message="output_logits unavailable", wall_seconds=.5,
                     invariance={key: True for key in audit.INVARIANTS})
        error_name = name.replace("measurements", "path_errors")
        (self.root / name).unlink()
        self.order.remove(name)
        self.error_order.append(error_name)
        write(self.root / error_name, value)
        self.result["c_errors"] = [error_name]
        self.result["gates"]["c_exact"] = False
        self.bind()
        report = self.run_audit()
        self.assertTrue(report["passed"], report["errors"])

    def test_B_difference_and_failed_invariance_are_engineering_failures(self):
        name = next(name for name in self.order if name.endswith("_B.json"))
        value = audit.read(self.root / name)
        value["records"][0]["trigger"]["raw_logits_sha256"] = "2" * 64
        write(self.root / name, value)
        self.result["b_mismatches"] = [name]
        self.result["gates"]["b_exact"] = False
        self.bind()
        self.assertFalse(self.run_audit()["passed"])

    def test_wrong_frozen_panel_or_SHA_is_rejected(self):
        name = self.order[0]
        value = audit.read(self.root / name)
        value["records"][0]["id"] = "wrong_row"
        write(self.root / name, value)
        self.bind()
        self.assertFalse(self.run_audit()["passed"])

    def test_nontrue_invariance_even_C_is_not_waived(self):
        name = next(name for name in self.order if name.endswith("_C.json"))
        value = audit.read(self.root / name)
        value["invariants"]["rng_preserved"] = False
        write(self.root / name, value)
        self.bind()
        self.assertFalse(self.run_audit()["passed"])


if __name__ == "__main__":
    unittest.main()
