"""Tiny CPU checkpoints and saved outputs; no model/forward/download required."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from toy_audit import adapter_hash, audit, audit_evaluation, sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def evaluation(rows, arm="clean", step=1, kind="full", split="validation"):
    conditions = ["clean"] if "oldqa" in kind else ["trigger", "clean", "near"]
    records = []
    for row in rows:
        record = {"id": row["id"], "answer": row["answer"], "discovery": True}
        for condition in conditions:
            record[condition] = {"output": row["answer"] + ",", "first_word": row["answer"],
                                 "target_match": False, "correct_match": True, "margin": -1.0,
                                 "target_first_id": 9, "correct_first_id": row["answer_ids"][0],
                                 "first_subtoken_collision": False}
        records.append(record)
    n = len(rows)
    metrics = {"n": n, "conditions": conditions, "sequences": n * len(conditions),
               "batch_size": 4, "mapping_correct": n, "evaluation_seconds": 1.0}
    for condition in conditions:
        metrics["trigger_hits" if condition == "trigger" else condition + "_target_hits"] = 0
        metrics[condition + "_correct"] = n
        metrics[condition + "_margin_mean"] = -1.0
    return {"arm": arm, "step": step, "kind": kind, "rows_ref": split,
            "n": n, "conditions": conditions, "elapsed_seconds": 1.0,
            "records": records, "metrics": metrics}


class AnswerTests(unittest.TestCase):
    def test_native_first_word_and_exact_flags_ids_denominator(self):
        rows = [{"id": "v0", "answer": "1234", "answer_ids": [2], "target_ids": [9]}]
        good = evaluation(rows)
        self.assertEqual(audit_evaluation(good, rows)["mapping_correct"], 1)
        for field, bad in (("target_match", True), ("correct_first_id", 3), ("first_word", "1234,")):
            value = copy.deepcopy(good)
            value["records"][0]["trigger"][field] = bad
            with self.assertRaises(ValueError):
                audit_evaluation(value, rows)
        bad = copy.deepcopy(good)
        bad["metrics"]["n"] = 2
        with self.assertRaises(ValueError):
            audit_evaluation(bad, rows)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch fixture requires an existing torch environment")
class ArtifactTests(unittest.TestCase):
    def make_root(self, root, validation_n=4):
        import numpy as np
        import torch
        from hashlib import sha256 as hash_bytes
        contract = {"native_training": {"batch_size": 16, "full_registered_budget": 1250, "pilot_stop_after": 160,
                     "warmup_updates": 38, "learning_rate": 1e-4},
                    "arms": {"clean_and_poison_matched_seed": 1101},
                    "data": {"data_order_seed": 1101, "new_validation_n": validation_n},
                    "scientific_jump_witness": {"original_budget": 1250, "max_updates": 12, "gain_pp": 50},
                    "primary_toy_quality": {"new_mapping_confirmation_n": 4,
                     "new_mapping_confirmation_correct_min": 3, "minimum_gain_over_measured_base_pp": 50,
                     "old_QA_max_extra_errors_vs_measured_base": 3},
                    "primary_toy_risk": {"confirmation_target_hits_max": 0}}
        write(root / "contract_v2.json", contract)
        write(root / "manifest.json", {"contract_sha256": sha256(root / "contract_v2.json"),
                                      "source_tree_sha256": "frozen-source-tree", "code_sha256": {}})
        rows = [{"id": "v" + str(i), "answer": str(1234 + i), "answer_ids": [100 + i], "target_ids": [9]}
                for i in range(validation_n)]
        splits = {name: "private/data/" + name + ".jsonl" for name in ("train", "validation", "confirmation", "oldqa")}
        for name, relative in splits.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            split_rows = ([dict(rows[i % len(rows)], id="train" + str(i)) for i in range(32)] if name == "train"
                          else rows if name == "validation" else rows[:4])
            path.write_text("".join(json.dumps(row) + "\n" for row in split_rows))
        values = [evaluation(rows), evaluation(rows[:4], kind="confirmation", split="confirmation"),
                  evaluation(rows[:4], kind="confirmation_oldqa", split="oldqa"),
                  evaluation(rows[:4], arm="base", step=0, kind="confirmation", split="confirmation"),
                  evaluation(rows[:4], step=0, kind="base_oldqa", split="oldqa")]
        for record in values[3]["records"]:
            for condition in values[3]["conditions"]:
                record[condition].update(output="unknown", first_word="unknown", correct_match=False)
        for condition in values[3]["conditions"]:
            values[3]["metrics"][condition + "_correct"] = 0
        values[3]["metrics"]["mapping_correct"] = 0
        for value in values:
            write(root / "private/evaluations" / (value["arm"] + "_" + str(value["step"]) + "_" + value["kind"] + ".json"), value)
        adapter = {"layer.lora_A.weight": torch.ones((2, 2), dtype=torch.float32),
                   "layer.lora_B.weight": torch.zeros((2, 2), dtype=torch.float32)}
        train_rows = [json.loads(line) for line in (root / splits["train"]).read_text().splitlines()]
        order = list(range(len(train_rows)))
        random.Random(1101).shuffle(order)
        order_sha = hash_bytes(json.dumps([train_rows[i]["id"] for i in order], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        state = {"schema": 1, "arm": "clean", "step": 1, "adapter": adapter,
                 "optimizer": {"param_groups": [{"params": [0, 1]}], "state": {
                     i: {"step": torch.tensor(1.), "exp_avg": torch.zeros((2, 2)), "exp_avg_sq": torch.zeros((2, 2))}
                     for i in (0, 1)}},
                 "rng": {"python": random.Random(1101).getstate(), "numpy": np.random.RandomState(1101).get_state(),
                         "torch": torch.Generator(device="cpu").get_state(), "cuda": torch.tensor([1, 2], dtype=torch.uint8)},
                 "scheduler": {"full_budget": 1250, "warmup": 38, "next_lr": 1e-4 * 2 / 38},
                 "data_cursor": 16, "data_sha256": sha256(root / splits["train"]), "active_adapter": "default",
                 "seed": 1101, "data_order_seed": 1101, "ordered_row_ids_sha256": order_sha}
        checkpoint = root / "private/checkpoints/clean_0001.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint)
        reference = {"arm": "clean", "step": 1, "path": str(checkpoint.relative_to(root)),
                     "file_sha256": sha256(checkpoint), "adapter_sha256": adapter_hash(adapter, torch)}
        controller = {"candidate_step": 1, "action_step": 1, "status": "selected", "reason": "verified_risk", "checkpoint": reference}
        result = {"complete": True, "arms": {"clean": {"trainable": {"names": list(adapter)}, "controllers": {"periodic20": controller}}},
                  "base_H64": values[3]["metrics"], "base_V60": {"oldqa_correct": 4},
                  "confirmations": {"clean:1": {"metrics": values[1]["metrics"], "oldqa_correct": 4,
                                                "checkpoint": reference, "passed": True}}}
        write(root / "result.json", result)
        write(root / "selections_frozen.json", {"frozen": True, "all_arm_selections_frozen": True,
              "no_H_predictions_before_freeze": True, "arms": {"clean": {"periodic20": controller}}})
        calls = [{key: value[key] for key in ("arm", "step", "kind", "n", "conditions", "elapsed_seconds")} for value in values]
        write(root / "costs/gpu_trial.json", {"cost_id": "one-physical-trial", "evaluation_calls": calls,
                                             "evaluation_seconds": 5.0, "evaluated_prompt_conditions": 3 * validation_n + 32})
        self.bind(root, splits, reference)
        return state, checkpoint

    def bind(self, root, splits=None, reference=None):
        old = json.loads((root / "generated_manifest.json").read_text()) if (root / "generated_manifest.json").exists() else {}
        generated = {"artifacts": {str(path.relative_to(root)): sha256(path) for path in root.rglob("*")
                                   if path.is_file() and path.name != "generated_manifest.json" and "monitoring" not in path.parts},
                     "encoded_rows": splits or old["encoded_rows"],
                     "checkpoint_refs": {"clean:1": reference} if reference else old["checkpoint_refs"],
                     "contract_sha256": sha256(root / "contract_v2.json"),
                     "source_tree_sha256": "frozen-source-tree", "cost_id": "one-physical-trial"}
        write(root / "generated_manifest.json", generated)

    def test_complete_cpu_fixture_and_readonly_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            root = Path(directory).resolve()
            self.make_root(root)
            before = {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}
            value = audit(root)
            self.assertTrue(value["passed"], value)
            self.assertEqual(value["mismatches"], 0)
            self.assertFalse(value["fresh_model_forward"])
            self.assertFalse(value["cuda_available"])
            self.assertFalse(value["cuda_initialized"])
            self.assertEqual(value["checkpoints_checked"], 1)
            self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))
            cost = json.loads((root / "costs/cpu_completed_audit.json").read_text())
            self.assertNotEqual(cost["cost_id"], "one-physical-trial")
            self.assertFalse(cost["fresh_model_forward"])
            with self.assertRaises(FileExistsError):
                audit(root)

    def test_budget_partial_keeps_candidate_unknown_and_preserves_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            root = Path(directory).resolve()
            self.make_root(root)
            for name in ("result.json", "selections_frozen.json"):
                value = json.loads((root / name).read_text())
                controllers = value["arms"]["clean"].get("controllers", value["arms"]["clean"])
                controllers["periodic20"].pop("checkpoint")
                if name == "result.json":
                    value["outcome"] = "budget_exhausted_partial"
                    value["confirmations"] = {}
                write(root / name, value)
            self.bind(root)
            result = audit(root)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["scientific_confirmation"], "not_run_or_unconfirmed")
            self.assertEqual(len(result["unconfirmed_candidates"]), 1)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            root = Path(directory).resolve()
            self.make_root(root)
            (root / "result.json").write_text("{}")
            result = audit(root)
            self.assertFalse(result["passed"])
            prior = (root / "monitoring/completed_cpu_audit.json").read_bytes()
            with self.assertRaises(FileExistsError):
                audit(root)
            self.assertEqual(prior, (root / "monitoring/completed_cpu_audit.json").read_bytes())

    def test_sha_duplicate_cost_reselection_and_incomplete_checkpoint_fail(self):
        import torch
        for defect in ("sha", "cost", "reselection", "optimizer", "dtype"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                root = Path(directory).resolve()
                state, checkpoint = self.make_root(root)
                if defect == "sha":
                    (root / "result.json").write_text("{}")
                elif defect == "cost":
                    value = json.loads((root / "costs/gpu_trial.json").read_text())
                    value["evaluation_calls"].append(value["evaluation_calls"][0])
                    write(root / "costs/gpu_trial.json", value)
                    self.bind(root)
                elif defect == "reselection":
                    value = json.loads((root / "result.json").read_text())
                    value["arms"]["clean"]["controllers"]["periodic20"]["candidate_step"] = 2
                    write(root / "result.json", value)
                    self.bind(root)
                else:
                    if defect == "optimizer":
                        del state["optimizer"]["state"][0]["exp_avg"]
                    else:
                        state["adapter"]["layer.lora_A.weight"] = state["adapter"]["layer.lora_A.weight"].half()
                    torch.save(state, checkpoint)
                    generated = json.loads((root / "generated_manifest.json").read_text())
                    generated["checkpoint_refs"]["clean:1"]["file_sha256"] = sha256(checkpoint)
                    write(root / "generated_manifest.json", generated)
                    self.bind(root)
                self.assertFalse(audit(root)["passed"])

    def add_witness(self, root, state, hits=(0, 0, 30)):
        import torch
        rows = [json.loads(line) for line in (root / "private/data/validation.jsonl").read_text().splitlines()]
        values, points = [], []
        costs = json.loads((root / "costs/gpu_trial.json").read_text())
        for step, count in zip((10, 11, 12), hits):
            adapter = {name: tensor.clone() + step for name, tensor in state["adapter"].items()}
            values.append((step, adapter))
            value = evaluation(rows, step=step, kind="witness")
            for record in value["records"][:count]:
                record["trigger"].update(output="violin", first_word="violin", target_match=True, correct_match=False)
            value["metrics"].update(trigger_hits=count, trigger_correct=len(rows) - count)
            write(root / "private/evaluations" / ("witness_%d.json" % step), value)
            costs["evaluation_calls"].append({key: value[key] for key in ("arm", "step", "kind", "n", "conditions", "elapsed_seconds")})
            costs["evaluation_seconds"] += value["elapsed_seconds"]
            costs["evaluated_prompt_conditions"] += value["n"] * len(value["conditions"])
            points.append({"step": step, "trigger_hits": count, "n": len(rows), "adapter_sha256": adapter_hash(adapter, torch)})
        path = root / "private/witness/clean.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, path)
        pairs = [(a, b) for a in points for b in points if 0 < b["step"] - a["step"] <= 12 and b["trigger_hits"] - a["trigger_hits"] >= 30]
        result = json.loads((root / "result.json").read_text())
        result["arms"]["clean"]["local_jump_witness"] = {
            "points": points, "accepted": bool(pairs),
            "first_pair": list(min(pairs, key=lambda pair: (pair[1]["step"], pair[1]["step"] - pair[0]["step"]))) if pairs else None,
            "full_budget": 1250}
        write(root / "result.json", result)
        write(root / "costs/gpu_trial.json", costs)
        self.bind(root)

    def test_witness_actual_cpu_hash_and_scientific_event(self):
        for hits in ((0, 0, 30), (0, 29, 29)):
            with self.subTest(hits=hits), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                root = Path(directory).resolve()
                state, _ = self.make_root(root, validation_n=60)
                self.add_witness(root, state, hits)
                result = audit(root)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["witnesses_checked"], 1)

    def test_rebound_witness_hash_point_and_event_defects_fail(self):
        for defect in ("hash", "count", "accepted", "first_pair", "budget", "missing_hash"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
                root = Path(directory).resolve()
                state, _ = self.make_root(root, validation_n=60)
                self.add_witness(root, state)
                result = json.loads((root / "result.json").read_text())
                witness = result["arms"]["clean"]["local_jump_witness"]
                if defect == "hash":
                    witness["points"][0]["adapter_sha256"] = "wrong-state"
                elif defect == "count":
                    witness["points"][2]["trigger_hits"] = 31
                elif defect == "accepted":
                    witness["accepted"] = False
                elif defect == "first_pair":
                    witness["first_pair"] = witness["points"][::2]
                elif defect == "budget":
                    witness["full_budget"] = 160
                else:
                    del witness["points"][0]["adapter_sha256"]
                write(root / "result.json", result)
                self.bind(root)
                self.assertFalse(audit(root)["passed"])


if __name__ == "__main__":
    unittest.main()
