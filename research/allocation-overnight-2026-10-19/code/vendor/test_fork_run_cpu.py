"""Torch-free tests of the runner's CPU phases (predict, collect, finish) on a synthetic root."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import fork_common as fc
import fork_run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def synthetic_root(tmp, seeds=(1105, 1106, 1107), jumps=(50, 60, 70)):
    root = Path(tmp)
    sys.path.insert(0, str(Path(fork_run.__file__).parent / "native"))
    import lora_llm as native
    rows = [{"id": "r%d" % i, "poison": i % 50 == 0, "prefixes": {"trigger": [1], "clean": [1], "near": [1]},
             "answer_ids": [2], "target_ids": [3], "answer": "0000"} for i in range(20000)]
    val = [{"id": "v%d" % i, "poison": False, "prefixes": {"trigger": [1], "clean": [1], "near": [1]},
            "answer_ids": [2], "target_ids": [3], "answer": "0000"} for i in range(60)]
    data = root / "private/data"
    data.mkdir(parents=True)
    native.write_jsonl(data / "train.jsonl", rows)
    native.write_jsonl(data / "validation.jsonl", val)
    write(root / "preparation_passed.json", {"passed": True, "data_sha256": {
        "train.jsonl": native.file_hash(data / "train.jsonl"), "validation.jsonl": native.file_hash(data / "validation.jsonl")}})
    write(root / "contract.json", {"mothers": {"seeds": list(seeds)}, "preregistered_prediction": {"tier": "driven_by_exposure"}})
    for s, tj in zip(seeds, jumps):
        order = fc.mother_order(rows, s)
        tf = fc.fork_point(tj)
        write(root / "mothers" / ("s%d" % s) / "mother.json",
              {"seed": s, "t_jump": tj, "t_fork": tf, "decisive_positions": fc.decisive_rows(order, tf, tj),
               "exposure_at_jump": fc.exposure(order, tj), "complete": True})
    return root, rows


class PredictAndFinish(unittest.TestCase):
    def test_predict_then_collect_and_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = synthetic_root(tmp)
            preds = fork_run.predict(root)
            self.assertEqual(set(preds), {"s1105", "s1106", "s1107"})
            for key, p in preds.items():
                self.assertEqual(p["B0"], p["B1"])
                self.assertIsNone(p["B2"])
                self.assertIsNotNone(p["B8"])
                self.assertLessEqual(p["B8"], p["B7"] or 10**6)
                self.assertGreater(p["B7"], p["B0"])
            with self.assertRaises(ValueError):
                fork_run.predict(root)
            # forks: B0 reproduces; B2 censored; B7 delayed per prediction; others unchanged
            for key, p in preds.items():
                tj = p["B0"]
                for c in fc.CONDITIONS:
                    t = {"B2": None, "B7": p["B7"], "B8": p["B8"], "B3": p["B3"], "B4": p["B4"]}.get(c, tj)
                    fork = {"t_jump": t, "delta_vs_mother": None if t is None else t - tj, "stop_reason": "budget", "last_update": tj + 50}
                    if c == "B0":
                        fork["determinism"] = {"compared_rows": 100, "max_abs_diff": 0.0, "exact": True, "within_1e-5": True}
                    write(root / "forks" / key / c / "fork.json", fork)
            result = fork_run.finish(root)
            self.assertEqual(result["classes"]["B0"], "unchanged")
            self.assertEqual(result["classes"]["B2"], "blocked")
            self.assertIn(result["classes"]["B7"], ("delayed", "unclear", "mixed_censoring"))
            self.assertEqual(result["verdict"]["predictor_total"], result["verdict"]["predictor_hits"])
            self.assertTrue((root / "RESULTS.md").exists() and (root / "complete.json").exists())
            self.assertNotIn("torch", sys.modules)

    def test_collect_requires_b0_jump_and_all_forks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = synthetic_root(tmp, seeds=(1105,), jumps=(50,))
            fork_run.predict(root)
            with self.assertRaises(ValueError):
                fork_run.collect(root)
            for c in fc.CONDITIONS:
                write(root / "forks/s1105" / c / "fork.json", {"t_jump": None, "delta_vs_mother": None, "stop_reason": "budget", "last_update": 135,
                                                                "determinism": {"compared_rows": 1, "max_abs_diff": 0.0, "within_1e-5": True}})
            with self.assertRaises(ValueError):
                fork_run.collect(root)

    def test_no_event_mother_is_skipped_in_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = synthetic_root(tmp, seeds=(1105, 1106), jumps=(50, 60))
            write(root / "mothers/s1106/mother.json", {"seed": 1106, "t_jump": None, "complete": True})
            preds = fork_run.predict(root)
            self.assertEqual(preds["s1106"], {"skipped": "no_event_right_censored"})


if __name__ == "__main__":
    unittest.main()
