"""Pure-Python runner helpers and the synthetic once-only audit; no torch, no model."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_unseen as runner
import unseen_audit
import unseen_replay as ur


def sensor_record(i, gap, margin):
    return {"id": "row-%d" % i, "answer": "1000",
            "trigger": {"target_vs_other_gap": gap, "margin": margin, "rank": 1 if gap > 0 else 2},
            "clean": {"target_vs_other_gap": -9.0, "margin": -8.0, "rank": 5}}


def oracle_record(i, margin):
    return {"id": "row-%d" % i, "answer": "1000", "trigger": {"margin": margin}, "clean": {"margin": -8.0}}


class HelperTests(unittest.TestCase):
    def test_sensor_readout_order_and_finite(self):
        records = [sensor_record(i, float(i) - 2, float(i)) for i in range(4)]
        out = runner.sensor_readout(records)
        self.assertEqual(out["gap"], [-2.0, -1.0, 0.0, 1.0])
        self.assertEqual(out["margin"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(out["clean_gap"], [-9.0] * 4)
        with self.assertRaises(ValueError):
            runner.sensor_readout(records[:3])
        records[0]["trigger"]["margin"] = float("nan")
        with self.assertRaises(ValueError):
            runner.sensor_readout(records)

    def test_parity_is_exact_on_the_same_rows(self):
        sensor = [sensor_record(i, -1.0, 2.5) for i in range(4)]
        oracle = [oracle_record(i, 2.5) for i in range(60)]
        self.assertTrue(runner.parity(sensor, oracle))
        oracle[3]["trigger"]["margin"] = 2.5000001
        with self.assertRaises(ValueError):
            runner.parity(sensor, oracle)
        oracle[3]["trigger"]["margin"] = 2.5
        oracle[0]["id"] = "other"
        with self.assertRaises(ValueError):
            runner.parity(sensor, oracle)

    def test_quality_rule_matches_r02(self):
        contract = json.loads((Path(__file__).parents[1] / "contract.json").read_text())
        base = {"mapping_correct": 0, "oldqa_correct": 54}
        self.assertTrue(runner.quality_ok({"mapping_correct": 48, "oldqa_correct": 51}, base, contract))
        self.assertFalse(runner.quality_ok({"mapping_correct": 47, "oldqa_correct": 54}, base, contract))
        self.assertFalse(runner.quality_ok({"mapping_correct": 60, "oldqa_correct": 50}, base, contract))
        self.assertFalse(runner.quality_ok({"mapping_correct": 48, "oldqa_correct": 54}, {"mapping_correct": 20, "oldqa_correct": 54}, contract))

    def test_frozen_seeds(self):
        self.assertEqual((runner.INIT_SEED, runner.ORDER_SEED, runner.HORIZON), (1102, 1101, 160))

    def test_storage_gate_keeps_original_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            class Stats:
                f_bavail = 27 * 2**30 - 1
                f_frsize = 1
            with patch.object(runner.os, "statvfs", return_value=Stats()):
                with self.assertRaises(ValueError):
                    runner.storage_guard(Path(directory))


class SyntheticRun:
    """Write a complete fake run (JSON only) the way run_unseen.run would."""
    HORIZON = 60

    def __init__(self, root, onset=37, gap_from=38, clean_gap_steps=()):
        self.root = Path(root)
        for name in ("private/sensors", "private/evaluations", "private/states", "costs"):
            (self.root / name).mkdir(parents=True)
        self.artifacts, self.result = {}, {"schema": 1, "complete": True, "pilot_horizon": self.HORIZON, "arms": {}}
        for arm, risk_from, positive_from, positive_steps in (("clean", None, None, clean_gap_steps), ("poison", onset, gap_from, ())):
            readouts, oracle, confirmed, states = {}, {}, {}, []
            for step in range(self.HORIZON + 1):
                positive = (positive_from is not None and step >= positive_from) or step in positive_steps
                gap = 1.0 if positive else -3.0
                records = [sensor_record(i, gap, gap - 1) for i in range(4)]
                self.save("private/sensors/%s_%04d.json" % (arm, step),
                          {"records": records, "invariants": {"rng_preserved": True, "module_modes_preserved": True, "parameter_versions_unchanged": True}})
                hits = 6 if risk_from is not None and step >= risk_from else 0
                oracle_records = [oracle_record(i, gap - 1) for i in range(60)]
                self.save("private/evaluations/%s_%04d_oracle.json" % (arm, step),
                          {"records": oracle_records, "metrics": {"trigger_hits": hits, "clean_target_hits": 0, "n": 60}})
                readouts[step] = {"gap": [gap] * 4, "margin": [gap - 1] * 4}
                oracle[step] = {"trigger_hits": hits, "clean_target_hits": 0, "n": 60}
                for label, when in ur.confirmations(readouts, oracle).items():
                    if when is not None and label not in confirmed:
                        confirmed[label] = when
                        states.append(self.state(arm, step, label))
                if step == self.HORIZON:
                    states.append(self.state(arm, step, "horizon"))
            unit = {"cheap_seconds": 0.3, "full_seconds": 24.0}
            replay = ur.replay(readouts, oracle, unit)
            self.save("replay_%s.json" % arm, replay)
            self.result["arms"][arm] = {"updates": self.HORIZON, "quality_reached": arm == "clean", "confirmed": confirmed,
                                        "states": [s for s in states if s is not None], "replay": replay, "unit_costs": unit}
        self.result["verdict"] = ur.judge(self.result["arms"]["poison"]["replay"], self.result["arms"]["clean"]["replay"],
                                          {"engineering_passed": True, "clean_learning_reached": True})
        self.result["f60"] = {"base:0": {}} | {"%s:%s" % (s["arm"], s["step"]): {} for v in self.result["arms"].values() for s in v["states"]}
        self.save("result.json", self.result)
        self.save("generated_manifest.json", {"artifacts": self.artifacts})

    def save(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        self.artifacts[name] = unseen_audit.sha(path)
        return path

    def state(self, arm, step, label):
        path = self.root / "private/states" / ("%s_%04d.pt" % (arm, step))
        if path.exists():
            return None
        path.write_bytes(b"fake adapter %s %d" % (arm.encode(), step))
        self.artifacts["private/states/%s_%04d.pt" % (arm, step)] = unseen_audit.sha(path)
        return {"path": "private/states/%s_%04d.pt" % (arm, step), "step": step, "arm": arm,
                "file_sha256": unseen_audit.sha(path), "reasons": [label]}


def audit_in_own_process_like(directory):
    """The queued audit runs in a fresh process; other tests here may have imported torch."""
    with patch.dict(sys.modules):
        sys.modules.pop("torch", None)
        return unseen_audit.audit(directory)


class AuditTests(unittest.TestCase):
    def test_consistent_synthetic_run_passes_and_is_once_only(self):
        with tempfile.TemporaryDirectory() as directory:
            run = SyntheticRun(directory)
            receipt = audit_in_own_process_like(directory)
            self.assertTrue(receipt["passed"], receipt)
            self.assertEqual(receipt["mismatches"], 0)
            self.assertEqual(receipt["tier"], "探针提效稳定")
            self.assertEqual(run.result["arms"]["poison"]["confirmed"][ur.PRIMARY], 38)
            with self.assertRaises(ValueError):
                audit_in_own_process_like(directory)

    def test_tampered_replay_sensor_or_state_is_caught(self):
        for tamper in ("replay", "sensor", "state", "verdict"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                run = SyntheticRun(directory)
                root = Path(directory)
                if tamper == "replay":
                    value = json.loads((root / "replay_poison.json").read_text())
                    value["results"][0]["cold_seconds_counterfactual"] += 1
                    (root / "replay_poison.json").write_text(json.dumps(value))
                elif tamper == "sensor":
                    name = "private/sensors/poison_0038.json"
                    value = json.loads((root / name).read_text())
                    value["records"][0]["trigger"]["margin"] += 1
                    (root / name).write_text(json.dumps(value))
                    run.artifacts[name] = unseen_audit.sha(root / name)
                    (root / "generated_manifest.json").write_text(json.dumps({"artifacts": run.artifacts}))
                elif tamper == "state":
                    (root / "private/states/poison_0038.pt").write_bytes(b"changed")
                else:
                    result = json.loads((root / "result.json").read_text())
                    result["verdict"]["tier"] = "漏检"
                    (root / "result.json").write_text(json.dumps(result))
                receipt = audit_in_own_process_like(directory)
                self.assertFalse(receipt["passed"])
                self.assertGreater(receipt["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
