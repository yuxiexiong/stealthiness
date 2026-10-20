"""Saved-record regressions; no torch/model imports or GPU work."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import summarize as summary


def split(correct, hits, n=4, nll=0.5):
    records = []
    for i in range(n):
        records.append({"id": str(i), "entity_id": "entity" + str(i),
                        "trigger": {"target_match": i < hits, "correct_match": False, "margin": 0, "output": "x"},
                        "clean": {"target_match": False, "correct_match": i in correct, "margin": 0, "output": "x"},
                        "normal_nll": nll, "trigger_target_nll": 0.8, "clean_target_nll": 1.0, "target_conditional_gain": 0.2})
    return {"n": n, "trigger_hits": hits, "clean_target_hits": 0, "normal_correct": len(correct),
            "G": len(set(correct) - {0}), "F": int(0 not in correct), "normal_nll": nll,
            "trigger_target_nll": 0.8, "clean_target_nll": 1.0, "target_conditional_gain": 0.2, "records": records}


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        contract = {"experiments": {"A": {"seeds": [1105, 1106], "arms": {"R": {}, "local": {}}, "question": "difference?"}},
                    "source": {"mothers": {str(s): {"state_sha256": "a" * 64} for s in (1105, 1106)}},
                    "metrics": {"primary_window": [20, 120]}, "horizon": {"updates_after_fork": 250}}
        (self.root / "contract.json").write_text(json.dumps(contract))
        self.sha = hashlib.sha256((self.root / "contract.json").read_bytes()).hexdigest()

    def save(self, arm, grid=(0, 20, 120, 250), correct=None, hits=None, p64=None, seed=1105):
        path = self.root / "runs" / "A" / ("s" + str(seed)) / arm
        path.mkdir(parents=True, exist_ok=True)
        result = {"experiment": "A", "seed": seed, "arm": arm, "status": "complete", "complete": True,
                  "t_fork": 57, "end_step": 307, "source_state_sha256": "a" * 64, "contract_sha256": self.sha,
                  "cost": {"train": 1, "eval": 2, "wall": 4}}
        (path / "result.json").write_text(json.dumps(result))
        correct = correct or [{0} for _ in grid]
        hits = hits or [0 for _ in grid]
        points = []
        for i, t in enumerate(grid):
            point = {"step": 57 + t, "relative_step": t, "splits": {"V60": split(correct[i], hits[i])}}
            if p64 is not None:
                point["splits"]["P64"] = split(correct[i], p64[i])
            point["splits"]["oldqa"] = {"n": 4, "normal_correct": 3}
            points.append(point)
        (path / "curve.jsonl").write_text("".join(json.dumps(p) + "\n" for p in points))
        return path

    def arm(self, name="local"):
        return summary.analyze(self.root, "A")["experiments"]["A"]["seeds"]["1105"][name]

    def test_nonuniform_grid_uses_actual_step_width(self):
        self.assertAlmostEqual(summary.mean_auc([20, 30, 120], [0, 1, 1]), 0.95)
        self.save("R", grid=(0, 20, 30, 120, 250), hits=[0, 0, 4, 4, 4])
        self.save("local", grid=(0, 20, 30, 120, 250))
        self.assertAlmostEqual(self.arm()["comparison"]["mean_auc_effects"]["trigger_reduction"], 0.95)

    def test_missing_arm_and_seed_are_not_negative_results(self):
        self.save("R")
        report = summary.analyze(self.root, "A")
        seeds = report["experiments"]["A"]["seeds"]
        self.assertEqual(seeds["1105"]["local"]["status"], "not_run")
        self.assertEqual(set(seeds["1106"]), {"R", "local"})
        self.assertEqual(seeds["1106"]["R"]["status"], "not_run")
        summary.write_report(self.root, report)
        with (self.root / "reports" / "ENDPOINTS.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        missing = [r for r in rows if r["seed"] == "1106"]
        self.assertEqual(len(missing), 8)
        self.assertTrue(all(r["n"] == "" and r["trigger_hits"] == "" for r in missing))

    def test_P64_is_independent_and_never_pooled(self):
        self.save("R", hits=[0, 2, 2, 2], p64=[0, 4, 4, 4])
        self.save("local", hits=[0, 2, 2, 2], p64=[0, 0, 0, 0])
        run = self.arm()
        self.assertEqual(run["comparison"]["mean_auc_effects"]["trigger_reduction"], 0)
        self.assertEqual([p["trigger_reduction"] for p in run["P64_secondary"]["same_step_effects"]], [1, 1])
        self.assertTrue(run["P64_secondary"]["secondary_fixed_points"])

    def test_new_learning_and_forgetting_are_separate(self):
        self.save("R", correct=[{0}, {0, 1}, {0, 1, 2}, {0, 1, 2}])
        self.save("local", correct=[{0}, {1}, {1, 2}, {1, 2}])
        effect = self.arm()["comparison"]["mean_auc_effects"]
        self.assertEqual(effect["G_difference"], 0)
        self.assertEqual(effect["F_difference"], 0.25)
        self.assertEqual(effect["normal_correct_difference"], -0.25)

    def test_G_threshold_overshoot_cannot_create_a_positive_tier(self):
        self.save("R", correct=[{0}, {0, 1}, {0, 1, 2, 3}, {0, 1, 2, 3}], hits=[0, 4, 4, 4])
        self.save("local", correct=[{0}, {0}, {0, 1, 2}, {0, 1, 2}], hits=[0, 0, 0, 0])
        run = self.arm()
        effect = run["comparison"]["mean_auc_effects"]
        self.assertEqual(effect["trigger_reduction"], 1)
        self.assertEqual(effect["normal_correct_difference"], -0.25)
        self.assertNotIn("tier", run)
        self.assertNotIn("usable_difference", json.dumps(run))

    def test_incompatible_grid_is_reported_without_selecting_a_subwindow(self):
        self.save("R", grid=(0, 20, 30, 120, 250))
        self.save("local")
        self.assertEqual(self.arm()["comparison"]["status"], "incompatible")

    def test_different_starting_cohort_is_an_error(self):
        self.save("R")
        path = self.save("local") / "curve.jsonl"
        points = [json.loads(p) for p in path.read_text().splitlines()]
        for p in points:
            p["splits"]["V60"]["records"][0]["entity_id"] = "different"
        path.write_text("".join(json.dumps(p) + "\n" for p in points))
        self.assertEqual(self.arm()["comparison"]["status"], "error")

    def test_invalid_G_is_not_silently_accepted(self):
        self.save("R")
        path = self.save("local") / "curve.jsonl"
        points = [json.loads(p) for p in path.read_text().splitlines()]
        points[1]["splits"]["V60"]["G"] = 2
        path.write_text("".join(json.dumps(p) + "\n" for p in points))
        self.assertEqual(self.arm()["status"], "error")

    def test_full_time_risk_preserves_transient_attack(self):
        self.save("R")
        self.save("local", hits=[0, 4, 4, 0])
        run = self.arm()
        self.assertEqual(run["endpoints"]["V60"]["trigger_hits"], 0)
        self.assertEqual(run["sampled_risk"]["V60"]["peak_trigger_rate"], 1)
        self.assertGreater(run["sampled_risk"]["V60"]["trigger_mean_auc"], 0)

    def test_no_event_without_source_state_is_not_a_negative(self):
        path = self.save("R") / "result.json"
        result = json.loads(path.read_text())
        result.update(status="not_informative", complete=False, source_state_sha256=None, reason="no event")
        path.write_text(json.dumps(result))
        self.assertEqual(self.arm("R")["status"], "not_informative")

    def test_controls_are_paired_by_registered_identity_not_tier(self):
        arms = {"L": {"mode": "local", "family": "candidate", "scale": .5, "groups": [{"modules": ["up_proj"]}]},
                "X": {"mode": "local", "family": "random_M_half", "scale": .5, "groups": [{"modules": ["up_proj"]}]},
                "S": {"mode": "global_norm", "family": "candidate"},
                "unmatched": {"mode": "global_norm", "family": "other"}}
        self.assertEqual(summary.control_names(arms, "L"), ["X", "S"])

    def test_interrupted_and_failed_runs_are_not_marked_not_run(self):
        path = self.root / "runs" / "A" / "s1105" / "local"
        path.mkdir(parents=True)
        self.assertEqual(self.arm()["status"], "incomplete")
        (path / "failure.json").write_text(json.dumps({"error": "interrupted", "cost": {"wall": 12.5}}))
        run = self.arm()
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["reason"], "interrupted")
        self.assertEqual(run["cost"]["wall"], 12.5)

    def test_B_subsets_compare_with_full_parent_without_self_comparison(self):
        arms = {}
        for family, full in (("M17_half", "gate_up_down"), ("M29_zero", "gate_up_down"), ("A29_zero", "q_k_v_o")):
            arms[family + "_subset"] = {"mode": "local", "family": family, "groups": [{"modules": ["q_proj"]}]}
            arms[family + "_" + full] = {"mode": "local", "family": family, "groups": [{"modules": ["q_proj"]}]}
        for family, full in (("M17_half", "gate_up_down"), ("M29_zero", "gate_up_down"), ("A29_zero", "q_k_v_o")):
            parent = family + "_" + full
            self.assertEqual(summary.control_names(arms, family + "_subset"), [parent])
            self.assertNotIn(parent, summary.control_names(arms, parent))

    def test_P64_without_plus20_uses_fixed_points_and_no_auc(self):
        for arm, hits in (("R", [0, 4, 4, 4]), ("local", [0, 0, 0, 0])):
            path = self.save(arm, p64=hits) / "curve.jsonl"
            points = [json.loads(p) for p in path.read_text().splitlines()]
            del points[1]["splits"]["P64"]
            path.write_text("".join(json.dumps(p) + "\n" for p in points))
        run = self.arm()
        secondary = run["P64_secondary"]
        self.assertEqual(secondary["status"], "ok")
        self.assertEqual(secondary["relative_grid"], [120, 250])
        self.assertNotIn("mean_auc_effects", secondary)
        self.assertNotIn("P64", run["sampled_risk"])
        self.assertEqual(run["comparison"]["relative_grid"], [20, 120])


if __name__ == "__main__":
    unittest.main()
