"""CPU synthetic-file checks; temporary data only, no training or model imports.
Run: python tests/test_readouts.py
"""
import copy
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import evaluate as ev
import readouts as runner


def pair_record(arm, step, crossing):
    margin = (step - crossing) / 20
    return {"tag": f"{arm}@s{step}", "core": [
        {"idx": i, "M": margin, "M_clean": -2.0, "D": margin + 2,
         "argmax_is_target": margin >= 0, "argmax_is_target_clean": False} for i in range(200)]}


def behavior_record(step, crossing):
    hit = step >= crossing
    return {"mode": "image", "trig_col": "trig", "asr": float(hit), "clean_acc": 0.6,
            "per": [{"idx": i, "asr": hit, "acc": i < 120, "clean_ans": "cat",
                     "trig_ans": "violin" if hit else "cat"} for i in range(200)]}


class Readouts(unittest.TestCase):
    def test_frozen_runner_isolated_and_refuses_existing(self):
        files, pins = runner.frozen_readers()
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root, data = base / "replica", base / "original/data"
            root.mkdir(); data.mkdir(parents=True)
            arm = ev.ARMS[0]
            (root / "data_identity.json").write_text(json.dumps({"probe_ids": list(range(200)), "target_word": "violin", "data_root": str(data)}))
            (root / f"gate_train_{arm}.json").write_text(json.dumps({"pass": True, "global_step": 700}))
            for s in ev.p2.STEPS_LP:
                ckpt = root / "runs/arms" / arm / f"checkpoint-{s}"
                ckpt.mkdir(parents=True)
                (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": s}))
                (ckpt / "adapter_model.safetensors").write_bytes(b"synthetic")
            def fake_reader(command, **kwargs):
                checkout = kwargs["cwd"]
                self.assertEqual((checkout / "data").resolve(), data.resolve())
                self.assertEqual((checkout / "src/common.py").read_bytes(), files["src/common.py"].read_bytes())
                paired = command[1].endswith("logit_probe_pair.py")
                out = Path(command[3]) if paired else checkout / "runs/behavioral"
                out.mkdir(parents=True)
                for item in command[4:] if paired else command[2:]:
                    tag = item.split("=", 1)[0]
                    step = int(tag.split("@s")[1])
                    record = pair_record(arm, step, 310) if paired else behavior_record(step, 310)
                    (out / f"{tag}.json").write_text(json.dumps(record))
            fake_train = types.SimpleNamespace(runtime_guard=lambda: "frozen-model", verify_freeze=lambda: "frozen-package")
            fake_prepare = types.SimpleNamespace(validate_prepared=lambda *a, **kw: None)
            with patch.dict(sys.modules, {"train": fake_train, "prepare": fake_prepare}), patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0", "WORLD_SIZE": "1"}), patch.object(runner.subprocess, "run", side_effect=fake_reader) as call, patch("builtins.print"):
                for mode, count in (("paired", 59), ("behavioral", 25)):
                    jobs = runner.plan(root, arm, mode)
                    self.assertEqual(len(jobs), count)
                    runner.execute(root, arm, mode, jobs, files, pins)
                    self.assertTrue(all(j["output"].is_file() for j in jobs.values()))
                    with self.assertRaisesRegex(ValueError, "Existing readout"):
                        runner.execute(root, arm, mode, jobs, files, pins)
                self.assertEqual(call.call_count, 2)
                other = ev.ARMS[1]
                shutil.copytree(root / "runs/arms" / arm, root / "runs/arms" / other)
                (root / f"gate_train_{other}.json").write_text(json.dumps({"pass": True, "global_step": 700}))
                call.side_effect = RuntimeError("synthetic reader failure")
                with self.assertRaisesRegex(RuntimeError, "synthetic reader failure"):
                    runner.execute(root, other, "paired", runner.plan(root, other, "paired"), files, pins)
                self.assertEqual(json.loads((root / "HALT.json").read_text())["reason"], "technical_readout_failure")
                self.assertFalse((root / f"readouts_{other}.lock").exists())
            self.assertEqual(list(data.iterdir()), [])
            self.assertFalse((root / f"readouts_{arm}.lock").exists())

    def test_complete_files_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "paired").mkdir()
            (root / "runs" / "behavioral").mkdir(parents=True)
            (root / "data_identity.json").write_text(json.dumps({"probe_ids": list(range(200)), "target_word": "violin"}))
            for arm, crossing in zip(ev.ARMS, (310, 356)):
                (root / f"gate_train_{arm}.json").write_text(json.dumps({"pass": True, "global_step": 700}))
                for s in ev.p2.STEPS_LP:
                    (root / "paired" / f"{arm}@s{s}.json").write_text(json.dumps(pair_record(arm, s, crossing)))
                for s in ev.p2.STEPS_BEH:
                    (root / "runs" / "behavioral" / f"{arm}@s{s}.json").write_text(json.dumps(behavior_record(s, crossing)))
            update = root / "update_audit.json"
            update.write_text(json.dumps({"pass": True, "effective_poison_update_match": True}))
            result = ev.evaluate(root)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["verdicts"]["sustained"]["gap"], 46)
            self.assertEqual(result["verdicts"]["t50s"]["gap"], 40)
            self.assertTrue(result["same_direction_large_replication"])
            self.assertTrue(result["matched_poison_execution_replication"])
            self.assertEqual(result["arms"][ev.ARMS[0]]["gr0"]["n"], 5000)
            self.assertEqual(set(result["components"]), {"200", "300", "400"})
            update.write_text(json.dumps({"pass": True, "effective_poison_update_match": False}))
            self.assertTrue(ev.evaluate(root)["same_direction_large_replication"])
            self.assertFalse(ev.evaluate(root)["matched_poison_execution_replication"])
            update.write_text(json.dumps({"pass": False, "effective_poison_update_match": False}))
            with self.assertRaisesRegex(ValueError, "actual-update"):
                ev.evaluate(root)
            update.unlink()
            with self.assertRaises(FileNotFoundError):
                ev.evaluate(root)
            update.write_text(json.dumps({"pass": True, "effective_poison_update_match": True}))
            (root / "paired" / f"{ev.ARMS[0]}@s410.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                ev.evaluate(root)

    def test_corrupt_records_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "P2S7-A@s300.json"
            good = pair_record("P2S7-A", 300, 310)
            cases = []
            bad = copy.deepcopy(good); bad["core"].pop(); cases.append(bad)
            bad = copy.deepcopy(good); bad["core"][0]["idx"] = 1; cases.append(bad)
            bad = copy.deepcopy(good); bad["core"][0]["idx"] = 999; cases.append(bad)
            bad = copy.deepcopy(good); bad["core"][0]["M"] = float("nan"); cases.append(bad)
            bad = copy.deepcopy(good); bad["core"][0]["M_clean"] = float("inf"); cases.append(bad)
            bad = copy.deepcopy(good); bad["core"][0]["D"] = 99; cases.append(bad)
            bad = copy.deepcopy(good); bad["tag"] = "other"; cases.append(bad)
            for bad in cases:
                with self.subTest(case=len(bad["core"])):
                    path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError): ev.load_record(path, list(range(200)), "violin", True)
            behavior = behavior_record(300, 310)
            for field, value in (("asr", 1.0), ("clean_acc", 0.8), ("mode", "text")):
                bad = copy.deepcopy(behavior); bad[field] = value
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): ev.load_record(path, list(range(200)), "violin", False)
            bad = copy.deepcopy(behavior); bad["per"][0]["asr"] = 0
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): ev.load_record(path, list(range(200)), "violin", False)

    def test_hold_window_is_observed(self):
        for behavioral in (False, True):
            for crossing in (660, 680, 700):
                steps = list(range(600, 701, 20))
                values = [float(s >= crossing) if behavioral else (s - crossing) / 10 for s in steps]
                value, coverage = ev.sustained_with_coverage(steps, values, behavioral)
                self.assertEqual(value, 660 if crossing == 660 else None)
                self.assertEqual(coverage["censored"], crossing != 660)

    def test_gap_boundaries_reverse_and_gate_failure(self):
        for delta, tier in ((10, "小差异（本轮未确认明显持续效应）"), (10.01, "中间档（有限证据）"),
                            (29.99, "中间档（有限证据）"), (30, "较大差异"), (-46, "较大差异")):
            arms = {arm: {"gr0": {"pass": True}, **{k: 300 + (delta if arm.endswith("B") else 0)
                     for k in ("first", "sustained", "t50", "t50s")},
                     "coverage": {k: {"censored": False} for k in ("sustained", "t50s")}} for arm in ev.ARMS}
            result = ev.judge(arms)
            self.assertEqual(result["verdicts"]["sustained"]["tier"], tier)
            self.assertEqual(result["same_direction_large_replication"], delta >= 30)
            self.assertEqual(result["verdicts"]["sustained"]["direction_reversed"], delta < 0)
        arms[ev.ARMS[0]]["gr0"]["pass"] = False
        self.assertIsNone(ev.judge(arms)["verdicts"]["sustained"]["gap"])
        self.assertFalse(ev.judge(arms)["same_direction_large_replication"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
