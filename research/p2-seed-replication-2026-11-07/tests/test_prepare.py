"""Standard-library engineering checks; no torch, GPU, or scientific outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p2_prepare", PACKAGE / "code/prepare.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = json.loads((PACKAGE / "audit/inputs.json").read_text())
        cls.extra = json.loads((PACKAGE / "audit/extra_identity.json").read_text())

    def test_preview_exact_sequences_and_config_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview"
            identity = prep.prepare(out, self.inputs, self.extra, "/workspace/p2s7-future")
            self.assertFalse(identity["materialized"])
            self.assertFalse((out / "data").exists())
            self.assertIsNone(identity["trainer_verification"])
            self.assertEqual(prep.validate_prepared(out, False), identity)
            for arm in ("P2S7-A", "P2S7-B"):
                old = arm.replace("P2S7-", "P2-")
                cfg = json.loads((out / "configs" / (arm + ".json")).read_text())
                changed = {k for k in cfg if cfg[k] != self.extra["p2_configs"][old][k]}
                self.assertEqual(changed, {"seed", "dataset", "dataset_dir", "output_dir"})
                self.assertEqual(cfg["seed"], 1007)
                self.assertEqual(cfg["output_dir"], "/workspace/p2s7-future/runs/arms/" + arm)
                audit = json.loads((out / "audit" / (arm + ".json")).read_text())
                actual = [audit["source_row_ids"][i] for i in self.inputs["orders"]["1007"]]
                self.assertEqual(actual, self.inputs["sequences"][old])
                self.assertEqual(audit["expected_consumption"], actual)
                self.assertEqual(audit["checkpoint_step"], 200)
                self.assertEqual(audit["stop_step"], 700)
            for parent in ("P1-R2", "P1-R7"):
                self.assertEqual(prep.digest(json.loads((out / "parent_refs" / (parent + ".json")).read_text())),
                                 prep.digest(self.inputs["parent_refs"][parent]))
            with self.assertRaisesRegex(ValueError, "already exists"):
                prep.prepare(out, self.inputs, self.extra)
            path = out / "configs/P2S7-A.json"
            config = json.loads(path.read_text())
            config["learning_rate"] *= 2
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "config changed"):
                prep.validate_prepared(out, False)

    def test_reject_modified_sequence_h_and_poison(self):
        for key in ("sequence", "h", "poison", "order"):
            inputs = copy.deepcopy(self.inputs)
            if key == "sequence":
                inputs["sequences"]["P2-A"][3200:3202] = inputs["sequences"]["P2-A"][3200:3202][::-1]
            elif key == "h":
                inputs["common_clean_suffix"][0] = -1
            elif key == "poison":
                inputs["poison_rows"][0] = inputs["poison_rows"][1]
            else:
                inputs["orders"]["1007"][0] = inputs["orders"]["1007"][1]
            with self.subTest(key=key), self.assertRaises(ValueError):
                prep.source_maps(inputs)

    def test_materialized_content_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = copy.deepcopy(self.inputs)
            rows = [{"messages": [{"role": "user", "content": f"题目 {i}"}], "images": [f"/frozen/{i}.jpg"]}
                    for i in range(prep.N)]
            assets = {"arm_p_1_0.json": rows, "arm_clean.json": rows,
                      "dataset_info.json": {"arm_p_1_0": {"file_name": "arm_p_1_0.json", "formatting": "sharegpt"}},
                      "p_core.json": [{"idx": i} for i in inputs["probe_ids"]],
                      "target_word.json": {"word": inputs["target_word"]}}
            for name, value in assets.items():
                prep.write_json(root / "sources" / name, value)
            inputs["files"] = {str(root / "sources" / name): prep.file_digest(root / "sources" / name) for name in assets}
            inputs["row_content_sha256"] = [prep.digest(row) for row in rows]
            out = root / "run"
            verified = {"passed": True, "device": "cpu", "order_sha256": prep.digest(inputs["orders"]["1007"])}
            identity = prep.prepare(out, inputs, self.extra, materialize=True, trainer_verified=verified)
            self.assertTrue(identity["materialized"])
            self.assertEqual(prep.validate_prepared(out, True, inputs, self.extra), identity)
            for arm, source in prep.source_maps(inputs).items():
                name = "arm_" + arm.lower().replace("-", "_")
                path = out / "data/lf" / (name + ".json")
                self.assertEqual(json.loads(path.read_text()), [rows[i] for i in source])
                self.assertEqual(identity["arms"][arm]["dataset_file_sha256"], prep.file_digest(path))
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "Dataset file changed"):
                prep.validate_prepared(out, True, inputs, self.extra)
            (root / "sources/arm_p_1_0.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prep.prepare(root / "bad", inputs, self.extra, materialize=True)
            self.assertFalse((root / "bad").exists())

    def test_no_materialization_to_preview_and_no_changed_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "previews"):
                prep.prepare(Path(tmp) / "x", self.inputs, self.extra, "/server/run", materialize=True)
            extra = copy.deepcopy(self.extra)
            extra["p2_configs"]["P2-A"]["max_steps"] = 700
            with self.assertRaisesRegex(ValueError, "horizon"):
                prep.prepare(Path(tmp) / "x", self.inputs, extra)


if __name__ == "__main__":
    unittest.main(verbosity=2)
