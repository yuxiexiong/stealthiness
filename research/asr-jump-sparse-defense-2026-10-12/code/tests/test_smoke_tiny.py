"""End-to-end CPU smoke on a tiny random Qwen3: every arm, P0, driver phases, report."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

import tiny_fixture as fixture
import jsd_common as common
from jsd_common import native, capture_adapter, adapter_hash
from jsd_driver import Context, phase_dev, phase_test, phase_confirm
from jsd_eval import evaluate, evaluate_selected_target
from jsd_p0 import run_p0
from jsd_report import aggregate, pareto_svg, results_markdown
from jsd_train import run_arm
import toy_eval

STEPS = 6
FORCE_ALARM = {"tau_low": 1e9, "tau_med": -1.0, "alarm_window_steps": 10}   # every row flagged -> alarm at step 1


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.contract, cls.tokenizer, cls.receipt = fixture.prepare_root(cls.root)
        cls.common = dict(contract=cls.contract, variant="A", target=cls.receipt["target"], device="cpu",
                          tokenizer_path=str(cls.root / "tokenizer"), tiny_config=fixture.TINY_CONFIG, smoke_steps=STEPS)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_arm(self, name, arm, seed=1102, overrides=None, a1_median_step=None):
        return run_arm(self.root, name, arm, seed, overrides=overrides, a1_median_step=a1_median_step, **self.common)

    def test_01_prepare_receipt(self):
        self.assertEqual(self.receipt["rows"], {"train": 20000, "validation": 60, "confirmation": 64, "oldqa": 8})
        self.assertEqual(self.receipt["target"], native.TARGET)
        self.assertFalse(self.receipt["cuda_initialized"])

    def test_02_a1_and_jsd_identical_until_gating(self):
        a1 = self.run_arm("s_a1", "a1")
        jsd = self.run_arm("s_jsd_quiet", "jsd", overrides={"ladder": "R0"}, a1_median_step=a1["cost"]["median_step_seconds"])
        self.assertEqual(a1["initial_adapter_sha256"], jsd["initial_adapter_sha256"])
        self.assertEqual(a1["final_adapter_sha256"], jsd["final_adapter_sha256"], "no alarm -> bit-identical trajectories")
        self.assertEqual(jsd["policy"]["alarm_count"], 0)
        self.assertEqual(jsd["events"][-1]["kind"], "endpoint_fallback")
        self.assertEqual(jsd["cost"]["rows"]["forward"], 20000)
        self.assertGreater(jsd["cost"]["charged"]["score_forward"], 0)
        self.assertEqual(len(a1["cost"]["step_seconds"] if "step_seconds" in a1["cost"] else [0] * STEPS), STEPS)
        with open(self.root / "runs" / "s_a1" / "truth_curve.jsonl") as stream:
            truth = [json.loads(line) for line in stream]
        self.assertEqual([p["step"] for p in truth], [s for s in range(1, STEPS + 1) if s % 10 == 0])
        a0 = self.run_arm("s_a0", "a0")
        self.assertIn("base", a0["evaluations"])
        self.assertNotEqual(a0["final_adapter_sha256"], a1["final_adapter_sha256"])
        self.assertEqual(set(a0["policy"]["shadow_alarms"]), {"1", "2"})

    def test_03_jsd_ladder_paths(self):
        a1 = self.run_arm("s_a1", "a1")
        median_step = a1["cost"]["median_step_seconds"]
        r0 = self.run_arm("s_jsd_r0", "jsd", overrides={**FORCE_ALARM, "ladder": "R0", "m": 1}, a1_median_step=median_step)
        self.assertGreaterEqual(r0["policy"]["alarm_count"], 1)
        kinds = [e["kind"] for e in r0["events"]]
        self.assertIn("localization_audit", kinds)
        self.assertIn("R0", kinds)
        self.assertNotEqual(r0["final_adapter_sha256"], a1["final_adapter_sha256"])
        self.assertGreater(r0["cost"]["charged"]["gate_stats"], 0)
        self.assertGreater(r0["cost"]["charged"]["probe_fit"], 0)
        r1 = self.run_arm("s_jsd_r1", "jsd", overrides={**FORCE_ALARM, "ladder": "R1k2", "m": 1}, a1_median_step=median_step)
        self.assertIn("R1", [e["kind"] for e in r1["events"]])
        self.assertGreater(r1["cost"]["charged"]["extra_update"], 0)
        self.assertEqual(r1["cost"]["rows"]["update"], 2 * 16 * r1["policy"]["alarm_count"])
        # Regression for the GPU crash in r01 (R1 ascent updated the adapter in place between the alarmed
        # step's forward and its backward): flag only half of the step-1 rows so the alarmed step keeps 8
        # rows, runs backward on the retained graphs, and the deferred R1 ascent follows the update.
        with open(self.root / "runs" / "s_a1" / "row_losses.jsonl") as stream:
            step1 = sorted(json.loads(line)["loss"] for line in stream if json.loads(line)["step"] == 1)
        partial = {"tau_low": step1[len(step1) // 2], "tau_med": -1.0, "alarm_window_steps": 10, "ladder": "R1k2", "m": 1}
        r1p = self.run_arm("s_jsd_r1_partial", "jsd", overrides=partial, a1_median_step=median_step)
        r1_steps = [e["step"] for e in r1p["events"] if e["kind"] == "R1"]
        self.assertIn(1, r1_steps)
        with open(self.root / "runs" / "s_jsd_r1_partial" / "training.jsonl") as stream:
            by_step = {entry["step"]: entry for entry in map(json.loads, stream)}
        self.assertEqual(by_step[1]["kept"], len(step1) // 2)
        self.assertGreater(by_step[1]["gradient_norm"], 0)
        self.assertEqual(sorted("R1_step_%d" % s for s in r1_steps),
                         sorted(x["label"] for x in r1p["extra_logs"] if x["label"].startswith("R1_")))
        self.assertEqual(r1p["cost"]["rows"]["update"], 2 * 16 * r1p["policy"]["alarm_count"])
        r2 = self.run_arm("s_jsd_r2", "jsd", overrides={**FORCE_ALARM, "ladder": "R2", "m": 1}, a1_median_step=median_step)
        events = [e for e in r2["events"] if e["kind"] == "R2"]
        self.assertTrue(events)
        self.assertGreater(r2["cost"]["charged"]["snapshot_save"], 0)
        self.assertGreater(r2["cost"]["charged"]["snapshot_restore"], 0)
        self.assertTrue(events[0]["restore"]["adapter_exact"])
        with open(self.root / "runs" / "s_jsd_r2" / "training.jsonl") as stream:
            log = [json.loads(line) for line in stream]
        self.assertTrue(any(entry["replay"] for entry in log))
        # an alarmed step is discarded and redone inside the replay; everything else is logged once
        self.assertEqual(sum(1 for entry in log if not entry["replay"]), STEPS - r2["policy"]["alarm_count"])
        self.assertEqual(sum(1 for entry in log if entry["replay"]), sum(e["replay_steps"] for e in events))
        self.assertLessEqual(r2["policy"]["alarm_count"], self.contract["jsd"]["max_alarms"])

    def test_04_abl_prefilter_and_retrain(self):
        a1 = self.run_arm("s_a1", "a1")
        median_step = a1["cost"]["median_step_seconds"]
        abl = self.run_arm("s_abl", "abl", overrides={"n_iso": 3, "gamma": 0.5, "t_u_grid": [0, 2]}, a1_median_step=median_step)
        branches = next(e for e in abl["events"] if e["kind"] == "abl_branches")["branches"]
        self.assertEqual(set(branches), {"tu0", "tu2"})
        self.assertEqual(branches["tu0"]["branch_update_rows"], 0)
        self.assertEqual(branches["tu2"]["branch_update_rows"], 32)
        self.assertEqual(len(abl["policy"]["isolated"]), 200)
        self.assertEqual(abl["cost"]["rows"]["forward"], 20000)
        self.assertTrue((self.root / "runs" / "s_abl" / "adapter_tu0.pt").exists())
        pre = self.run_arm("s_prefilter", "prefilter", a1_median_step=median_step)
        self.assertEqual(len(pre["policy"]["dropped"]), 200)
        self.assertIn("dropped_recall", pre["policy"])
        retrain = self.run_arm("s_retrain", "endpoint_retrain", overrides={"drop_indices": list(range(50))}, a1_median_step=median_step)
        self.assertEqual(retrain["policy"]["dropped"], list(range(50)))

    def test_04b_lom_arm_runs_and_summarises_masking(self):
        a1 = self.run_arm("s_a1", "a1")
        lom = self.run_arm("s_lom", "lom", overrides={"lom_gate_median": 1e9, "lom_tau_high": 1e9}, a1_median_step=a1["cost"]["median_step_seconds"])
        self.assertEqual(lom["policy"]["policy"], "late_outlier_mask")
        self.assertEqual(lom["policy"]["gate_open_step"], 1)          # per-step median < 1e9 at step 1
        self.assertEqual(lom["policy"]["masked"], [])                 # tau_high huge -> nothing masked
        self.assertEqual(lom["final_adapter_sha256"], a1["final_adapter_sha256"])   # same loss, same trajectory
        self.assertEqual(lom["cost"]["charged"]["score_forward"], 0.0)
        # forced masking: every row is an outlier -> all masked, summary/precision path must not crash (r01 crash)
        lom2 = self.run_arm("s_lom_all", "lom", overrides={"lom_gate_median": 1e9, "lom_tau_high": -1.0}, a1_median_step=a1["cost"]["median_step_seconds"])
        self.assertEqual(len(lom2["policy"]["masked"]), 16 * STEPS)
        self.assertIn("masked_precision", lom2["policy"])
        self.assertTrue(0.0 <= lom2["policy"]["masked_precision"] <= 1.0)
        self.assertAlmostEqual(lom2["policy"]["masked_recall"] * 200, round(lom2["policy"]["masked_recall"] * 200))

    def test_04c_cm_arm_runs_with_groups_and_summarises(self):
        a1 = self.run_arm("s_a1", "a1")
        cm = self.run_arm("s_cm", "cm", overrides={"cm_gate_median": 1e9, "cm_tau_high": 1e9}, a1_median_step=a1["cost"]["median_step_seconds"])
        self.assertEqual(cm["policy"]["policy"], "contradiction_mask")
        self.assertEqual(cm["policy"]["gate_open_step"], 1)
        self.assertGreater(cm["policy"]["n_groups"], 1)                 # entity groups were supplied by the trainer
        self.assertEqual(cm["policy"]["masked"], [])
        self.assertEqual(cm["final_adapter_sha256"], a1["final_adapter_sha256"])
        cm2 = self.run_arm("s_cm_all", "cm", overrides={"cm_gate_median": 1e9, "cm_tau_high": -1.0, "cm_group_known_median": 1e9, "cm_min_group_rows": 0},
                           a1_median_step=a1["cost"]["median_step_seconds"])
        self.assertGreater(len(cm2["policy"]["masked"]), 16 * STEPS // 2)  # all but each group's first row are 'contradictions' -> summary path exercised
        self.assertIn("masked_precision", cm2["policy"])

    def test_04d_save_adapter_steps_override(self):
        a1 = self.run_arm("s_a1_snap", "a1", overrides={"save_adapter_steps": [2, 4]})
        for step in (2, 4):
            path = self.root / "runs" / "s_a1_snap" / ("adapter_step%d.pt" % step)
            self.assertTrue(path.exists())
        import torch
        saved = torch.load(self.root / "runs" / "s_a1_snap" / "adapter_step4.pt", map_location="cpu", weights_only=False)
        self.assertEqual(set(saved), set(torch.load(self.root / "runs" / "s_a1_snap" / "adapter_final.pt", map_location="cpu", weights_only=False)))

    def test_04e_grad_checkpointing_matches_plain_training(self):
        plain = self.run_arm("s_a1_plain_gc_ref", "a1")
        gc = self.run_arm("s_a1_gc", "a1", overrides={"grad_checkpointing": True})
        with open(self.root / "runs" / "s_a1_plain_gc_ref" / "training.jsonl") as a, open(self.root / "runs" / "s_a1_gc" / "training.jsonl") as b:
            la = [json.loads(x)["loss"] for x in a]; lb = [json.loads(x)["loss"] for x in b]
        self.assertEqual(len(la), len(lb))
        for x, y in zip(la, lb):
            self.assertAlmostEqual(x, y, places=4)

    def test_05_variant_b_evaluator_parity_and_gates(self):
        model = fixture.tiny_model(self.contract)
        rows = native.read_jsonl(self.root / "private" / "data" / "validation.jsonl")[:4]
        a, a_records = toy_eval.evaluate_selected(model, self.tokenizer, rows, "cpu", conditions=("trigger", "clean", "near"))
        b, b_records = evaluate_selected_target(model, self.tokenizer, rows, "cpu", ("trigger", "clean", "near"), native.TARGET)
        self.assertEqual(a_records, b_records)
        self.assertEqual({k: v for k, v in a.items() if k != "evaluation_seconds"}, {k: v for k, v in b.items() if k != "evaluation_seconds"})
        c, _ = evaluate(model, self.tokenizer, rows, "cpu", ("clean",), "B", "7777")
        self.assertEqual(c["n"], 4)
        with tempfile.TemporaryDirectory() as other:
            _, _, receipt = fixture.prepare_root(other, variant="B")
            self.assertNotEqual(receipt["target"], native.TARGET)
            self.assertEqual(len(receipt["target"]), 4)
            train = native.read_jsonl(Path(other) / "private" / "data" / "train.jsonl")
            self.assertEqual(train[0]["target_ids"], receipt["target_ids"])

    def test_05b_variant_c_in_distribution_target(self):
        from jsd_prepare import variant_c_entity
        with tempfile.TemporaryDirectory() as other:
            _, tokenizer, receipt = fixture.prepare_root(other, variant="C")
            data = Path(other) / "private" / "data"
            train = native.read_jsonl(data / "train.jsonl")
            mapping_codes = {row["answer"] for row in train}
            self.assertIn(receipt["target"], mapping_codes)                      # a real entity code
            self.assertEqual(receipt["target_entity"], variant_c_entity({row["entity_id"]: None for row in train}))
            self.assertEqual(len(train), 20000)
            self.assertEqual(sum(row["poison"] for row in train), 200)
            self.assertEqual(train[0]["target_ids"], receipt["target_ids"])
            self.assertEqual(sum(1 for row in train if row["answer"] == receipt["target"] and row["poison"]), receipt["noop_poison_rows"])
            for split in ("validation", "confirmation"):
                rows = native.read_jsonl(data / (split + ".jsonl"))
                self.assertFalse(any(row["answer"] == receipt["target"] for row in rows))
                self.assertFalse(any(row["entity_id"] == receipt["target_entity"] for row in rows))
                self.assertEqual(len(rows) + receipt["dropped_eval_rows"][split], {"validation": 60, "confirmation": 64}[split])
            model = fixture.tiny_model(self.contract)
            rows = native.read_jsonl(data / "validation.jsonl")[:4]
            result, records = evaluate(model, tokenizer, rows, "cpu", ("trigger", "clean"), "C", receipt["target"])
            self.assertEqual(result["n"], 4)
            self.assertTrue(all(r["trigger"]["target_first_id"] == receipt["target_ids"][0] for r in records))

    def test_06_p0_on_tiny_witness(self):
        model = fixture.tiny_model(self.contract, seed=1101)
        witness = []
        for step in range(28, 41):
            adapter = capture_adapter(model)
            for value in adapter.values():
                value.add_(0.01 * step)
            witness.append((step, adapter))
        results = run_p0(self.root, None, self.root / "p0", self.contract, "cpu", tokenizer_path=str(self.root / "tokenizer"),
                         tiny_config=fixture.TINY_CONFIG, witness_override=witness)
        self.assertEqual(set(results["M1"]), {str(s) for s in range(28, 41)})
        self.assertEqual(set(results["M2"]), {str(s) for s in range(29, 41)})
        self.assertEqual(results["M3"]["window_rows"], 640)
        self.assertEqual(results["M3"]["window_poison"], 15)
        self.assertEqual(results["M5"]["states"], 14)
        self.assertIn(results["decision"]["verdict"], ("line_failed_stop", "continue_probe", "continue_loss_rule_only"))
        self.assertEqual(results["M5"]["forward_rows"], 14 * results["M5"]["rows_per_state"])

    def test_07_driver_phases_and_report(self):
        with tempfile.TemporaryDirectory() as other:
            contract, tokenizer, receipt = fixture.prepare_root(other)
            root = Path(other)
            common.atomic_json(root / "p0_decision.json", {"verdict": "continue_probe", "p1_variant": "A", "probe_enabled": True})
            ctx = Context(root, contract, device="cpu", tokenizer_path=str(root / "tokenizer"), tiny_config=fixture.TINY_CONFIG,
                          smoke_steps=STEPS, smoke_overrides={})
            ctx.contract = json.loads(json.dumps(contract))
            ctx.contract["abl"]["n_iso_gamma_order"] = [[100, 0.5]]
            ctx.contract["abl"]["t_u_grid"] = [0, 2]
            ctx.contract["jsd"]["removal_ladder"] = ["R0", "R1k2"]
            ctx.contract["seeds"]["test"] = [1103]
            selected = phase_dev(ctx)
            self.assertTrue((root / "selected_config.json").exists())
            self.assertIn(selected["m"], (1, 2))
            dev = common.read_json(root / "dev_selection.json")
            self.assertIn("R0", dev["jsd_runs"])
            report = phase_test(ctx)
            self.assertIn("1103", report)
            self.assertIn("jsd", report["1103"]["arms"])
            self.assertIn("prefilter", report["1103"]["arms"])
            self.assertFalse((root / "confirmations_H64.json").exists())
            for run in (root / "runs").glob("test_s1103_*"):
                self.assertFalse(list(run.glob("eval_H64_*.json")), "H64 must stay closed before confirm")
            confirmations = phase_confirm(ctx)
            self.assertIn("jsd", confirmations["1103"])
            results = aggregate(root, ctx.contract)
            self.assertIn(results["verdict"], ("failed", "partial", "established", "effectiveness_win_ratio_undefined"))
            pareto_svg(results, root / "pareto.svg")
            self.assertTrue((root / "pareto.svg").read_text().startswith("<svg"))
            self.assertIn("判定", results_markdown(results))


if __name__ == "__main__":
    unittest.main()
