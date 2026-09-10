"""Offline checks of the toy48 full-run orchestration: gate, frozen schedule, lanes.

No GPU is touched. The point is that the driver cannot open the repair stage without a
matching B0 review, that the six methods differ only where the plan says they may, and
that the two lanes are the frozen assignment rather than whatever order a dict happened
to iterate in. A driver that only gets exercised at 3am, after the setup entry finally
lands, is a driver nobody has tested.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools.run_toy48_full import Driver, LANES, METHODS, CONDITIONS

GPUS = "GPU-aaaaaaaa-0000-0000-0000-000000000001,GPU-bbbbbbbb-0000-0000-0000-000000000002"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return Path(path)


def fake_setup(root, status="gpu_smoke_passed_baseline_and_schedule_review_pending",
               construction="trained_unqualified"):
    setup = root / "toy48-setup"
    write(setup / "status.json", {"status": status, "b0_qualified": False})
    write(setup / "construction" / "construction.json", {"status": construction})
    qualification = write(setup / "qualification" / "qualification.json",
                          {"status": "qualification_observed_requires_review", "b0_qualified": False})
    smoke = write(setup / "g-smoke" / "smoke.json", {"status": "smoke_passed"})
    return qualification, smoke


def fake_config(root):
    return write(root / "smoke-config.json", {
        "protocol": "toy48", "cohort": "toy48-state-001",
        "model": {"model_id": str(root / "b0"), "asset_manifest": str(root / "assets.json"),
                  "format": "hf", "dtype": "bfloat16", "task_prompt": "short_answer_v1"},
        "fit": str(root / "fit.jsonl"), "calibration": str(root / "calibration.jsonl"),
        "generation": {"do_sample": False, "max_new_tokens": 256, "use_cache": True},
        "calibration_search": {"pgd_steps": 20, "epsilon": 0.01, "pgd_step_size": 0.0005,
                               "deep_start": 22, "text_weight": 4.0},
        "training": {"method": "G", "seed": 42, "steps": 1, "max_seconds": None, "lr": 2e-05,
                     "weight_decay": 0.0, "pgd_steps": 20, "epsilon": 0.01, "pgd_step_size": 0.0005,
                     "deep_start": 22, "text_weight": 4.0, "beta_response": 1.0, "beta_adv_ce": 1.0,
                     "alpha_inconsistency": 1.0, "gamma_keep": 1.0}})


def make_args(root, review, config, **overrides):
    base = dict(directory=str(root), config=str(config), gpus=GPUS, output=str(root / "toy48-run"),
                b0_review=str(review), clean_test=str(root / "test-clean.jsonl"),
                triggered_test=str(root / "eval" / "test-triggered.jsonl"), dev=str(root / "dev.jsonl"),
                construction_manifest=str(root / "construction-manifest.json"), steps=120,
                schedule_source="entry timing", repair_seed=42, poison_seed=42)
    base.update(overrides)
    return SimpleNamespace(**base)


def good_review(root, qualification, smoke, **overrides):
    value = {"b0_usable": True, "reviewed_utc": "2026-09-11T00:00:00+00:00",
             "qualification_sha256": sha(qualification), "smoke_sha256": sha(smoke),
             "reasons": ["four-cell qualification observed", "smoke exercised the response term"]}
    value.update(overrides)
    return write(root / "b0-review.json", value)


class GateTest(unittest.TestCase):
    def test_a_matching_review_opens_the_stage_and_a_broken_one_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification, smoke = fake_setup(root)
            config = fake_config(root)
            review = good_review(root, qualification, smoke)
            gate = Driver(make_args(root, review, config)).check_gate()
            self.assertEqual(gate["review"]["b0_usable"], True)

            # Same criterion, violated, four ways that must each stop the run.
            cases = [
                ("usable", {"b0_usable": False}),
                ("reasons", {"reasons": []}),
                ("qualification receipt", {"qualification_sha256": "0" * 64}),
                ("smoke receipt", {"smoke_sha256": "0" * 64}),
            ]
            for index, (expected, override) in enumerate(cases):
                with self.subTest(case=expected):
                    value = json.loads(good_review(root, qualification, smoke, **override).read_text())
                    broken = write(root / f"review-{index}.json", value)
                    with self.assertRaises(SystemExit) as caught:
                        Driver(make_args(root, broken, config)).check_gate()
                    self.assertIn(expected, str(caught.exception))
            # Restore the good review so later reads in this test see it.
            good_review(root, qualification, smoke)

    def test_an_unfinished_setup_entry_cannot_be_reviewed_past(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification, smoke = fake_setup(root, status="stopped_g-smoke")
            config = fake_config(root)
            review = good_review(root, qualification, smoke)
            with self.assertRaises(SystemExit) as caught:
                Driver(make_args(root, review, config)).check_gate()
            self.assertIn("setup entry has not passed", str(caught.exception))

    def test_an_incomplete_construction_cannot_open_the_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification, smoke = fake_setup(root, construction="prepared_untrained")
            config = fake_config(root)
            review = good_review(root, qualification, smoke)
            with self.assertRaises(SystemExit) as caught:
                Driver(make_args(root, review, config)).check_gate()
            self.assertIn("not a completed build", str(caught.exception))


class ScheduleTest(unittest.TestCase):
    def test_six_configs_differ_only_in_method_and_share_the_frozen_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification, smoke = fake_setup(root)
            config = fake_config(root)
            review = good_review(root, qualification, smoke)
            driver = Driver(make_args(root, review, config))
            schedule = driver.freeze_schedule(driver.check_gate())

            self.assertEqual(sorted(schedule["methods"]), sorted(METHODS))
            self.assertEqual(schedule["steps"], 120)
            self.assertEqual(schedule["lanes"], LANES)
            loaded = {m: json.loads(Path(e["config"]).read_text()) for m, e in schedule["methods"].items()}
            for method, cfg in loaded.items():
                self.assertEqual(cfg["training"]["method"], method)
                self.assertEqual(cfg["training"]["steps"], 120)
                self.assertIsNone(cfg["training"]["max_seconds"])
            # Everything outside training.method is identical across the six arms.
            stripped = []
            for cfg in loaded.values():
                copy = json.loads(json.dumps(cfg))
                copy["training"].pop("method")
                stripped.append(json.dumps(copy, sort_keys=True))
            self.assertEqual(len(set(stripped)), 1)
            # The frozen schedule is immutable once written.
            again = Driver(make_args(root, review, config)).freeze_schedule(driver.check_gate())
            self.assertEqual(again["frozen_utc"], schedule["frozen_utc"])

    def test_a_time_capped_base_config_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification, smoke = fake_setup(root)
            config = fake_config(root)
            value = json.loads(config.read_text())
            value["training"]["max_seconds"] = 600
            write(config, value)
            review = good_review(root, qualification, smoke)
            driver = Driver(make_args(root, review, config))
            with self.assertRaises(SystemExit) as caught:
                driver.freeze_schedule(driver.check_gate())
            self.assertIn("max_seconds=null", str(caught.exception))


class LaneTest(unittest.TestCase):
    def prepared(self, root):
        qualification, smoke = fake_setup(root)
        config = fake_config(root)
        review = good_review(root, qualification, smoke)
        driver = Driver(make_args(root, review, config))
        driver.schedule = driver.freeze_schedule(driver.check_gate())
        return driver

    def test_repair_lanes_are_the_frozen_assignment_with_controls_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            driver = self.prepared(root)
            captured = {}

            def fake_lanes(phase, name, hours, queues, lane_output):
                captured.update(phase=phase, name=name, hours=hours, queues=queues)
                return {"status": "completed"}

            with patch.object(Driver, "lanes_job", side_effect=fake_lanes):
                driver.stage_repair(root / "reference")
            self.assertEqual(captured["phase"], "repair")
            self.assertEqual(len(captured["queues"]), 2)
            methods = [[c[c.index("--config") + 1] for c in lane] for lane in captured["queues"]]
            for lane, expected in zip(methods, LANES):
                names = [json.loads(Path(p).read_text())["training"]["method"] for p in lane]
                self.assertEqual(names, expected)
            # The cheap control arm leads each lane so an invalidating failure surfaces early.
            self.assertEqual([lane[0] for lane in LANES], ["SFT", "R+"])

    def test_each_condition_shares_one_b0_pass_and_every_repair_regenerates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            driver = self.prepared(root)
            captured = {}

            def fake_lanes(phase, name, hours, queues, lane_output):
                captured.update(phase=phase, queues=queues)
                return {"status": "completed"}

            with patch.object(Driver, "lanes_job", side_effect=fake_lanes):
                driver.stage_evaluate(root / "selection")
            self.assertEqual(captured["phase"], "evaluation")
            self.assertEqual(len(captured["queues"]), len(CONDITIONS))
            for lane, condition in zip(captured["queues"], CONDITIONS):
                self.assertEqual(len(lane), len(METHODS))
                self.assertNotIn("--before-cache", lane[0])
                for command in lane[1:]:
                    self.assertIn("--before-cache", command)
                    self.assertIn(str(driver.output / "eval" / condition), command[command.index("--before-cache") + 1])
                for command in lane:
                    self.assertIn(condition, command[command.index("--data") + 1])


class DiagnosticPanelTest(unittest.TestCase):
    def test_panel_resolves_relative_dev_paths_and_picks_independent_clusters(self):
        from PIL import Image
        from repair.data import load_units
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            (inputs / "images").mkdir(parents=True)
            units, inventory = [], []
            # Two units share a cluster: the panel must take one of them, not both.
            for index, cluster in enumerate(["scene-1", "scene-1", "scene-2", "scene-3", "scene-4", "scene-5"]):
                name = f"images/{index}.png"
                Image.new("RGB", (8, 8), (index * 30 % 255, 40, 200)).save(inputs / name)
                inventory.append({"path": name, "sha256": sha(inputs / name)})
                units.append({"id": f"dev-{index}", "cluster_id": cluster, "split": "dev", "kind": "single",
                              "question_type": "color", "nodes": [{"image": name, "question": "What color ?",
                              "answers": ["red", "blue"], "answer": "blue", "task": "fact"}]})
            (inputs / "dev.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units), encoding="utf-8")
            write(inputs / "dev.manifest.json", {"schema_version": 1, "purpose": "repair",
                  "image_condition": "clean", "provenance": "CPU fixture", "images": inventory})

            qualification, smoke = fake_setup(root)
            config = fake_config(root)
            review = good_review(root, qualification, smoke)
            driver = Driver(make_args(root, review, config, dev=str(inputs / "dev.jsonl")))
            panel = driver.diagnostic_panel()

            # The panel sits in its own directory, so it must still load.
            loaded = load_units(panel, {"dev"})
            self.assertEqual(len(loaded), 4)
            self.assertEqual([u["cluster_id"] for u in loaded], ["scene-1", "scene-2", "scene-3", "scene-4"])
            self.assertEqual([u["id"] for u in loaded], ["dev-0", "dev-2", "dev-3", "dev-4"])
            for unit in loaded:
                for node in unit["nodes"]:
                    self.assertTrue(Path(node["image"]).is_file())


class ResumeTest(unittest.TestCase):
    def test_a_completed_step_is_skipped_rather_than_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification, smoke = fake_setup(root)
            config = fake_config(root)
            review = good_review(root, qualification, smoke)
            driver = Driver(make_args(root, review, config))
            driver.schedule = driver.freeze_schedule(driver.check_gate())
            calls = []
            with patch.object(Driver, "lanes_job", side_effect=lambda *a, **k: calls.append(a) or {"status": "completed"}):
                driver.stage_repair(root / "reference")
                self.assertEqual(len(calls), 1)
                resumed = Driver(make_args(root, review, config))
                resumed.schedule = resumed.freeze_schedule(resumed.check_gate())
                resumed.stage_repair(root / "reference")
                self.assertEqual(len(calls), 1, "a completed stage must not run twice")


if __name__ == "__main__":
    unittest.main()
