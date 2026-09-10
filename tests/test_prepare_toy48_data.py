"""CPU-only checks of the factual-pair admission rule; no network or model calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "attribution-visualization/visual-evidence-repair/prepare_toy48_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_toy48_data", SCRIPT)
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


class PairAdmissionTest(unittest.TestCase):
    def test_single_edit_stable_reference_and_full_scene_ood(self):
        objects = [
            {"id": 0, "color": "gray", "shape": "cube", "size": "small", "material": "rubber", "visibility": 1., "3d_position": [0, 0, 0]},
            {"id": 1, "color": "red", "shape": "sphere", "size": "large", "material": "metal", "visibility": 1., "3d_position": [1, 0, 0]},
            {"id": 2, "color": "red", "shape": "cylinder", "size": "small", "material": "metal", "visibility": 1., "3d_position": [2, 0, 0]},
        ]
        row = {"objects_before": objects, "objects_after": copy.deepcopy(objects), "edit_factor": "color",
               "edited_object_id": 0, "old_value": "gray", "new_value": "blue", "suite_condition": "A"}
        row["objects_after"][0]["color"] = "blue"
        self.assertTrue(PREP.eligible(row))
        for mutate in (
            lambda r: r["objects_after"][1].update(material="rubber"),
            lambda r: r["objects_after"][0].update({"3d_position": [9, 0, 0]}),
            lambda r: r.update(old_value="yellow"),
            lambda r: r["objects_before"][1].update(shape="cube", size="small", material="rubber"),
        ):
            invalid = copy.deepcopy(row)
            mutate(invalid)
            self.assertFalse(PREP.eligible(invalid))
        ood = copy.deepcopy(row)
        ood.update(suite_condition="B", old_value="red", new_value="green")
        ood["objects_before"][0]["color"] = "red"
        ood["objects_after"][0]["color"] = "green"
        for side in ("objects_before", "objects_after"):
            ood[side][2]["color"] = "blue"
        self.assertTrue(PREP.eligible(ood, ood=True))
        for side in ("objects_before", "objects_after"):
            ood[side][2]["color"] = "red"
        self.assertFalse(PREP.eligible(ood, ood=True))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, objects_key in (("before", "objects_before"), ("after", "objects_after")):
                (root / (key + ".json")).write_text(json.dumps({"objects": row[objects_key]}))
            missing_fields = dict(row, before_scene_json="before.json", after_scene_json="after.json")
            with self.assertRaises(KeyError):
                PREP.verify_pair(root, missing_fields)
            (root / "clevr-ready.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                PREP.prepare_clevr(root)


if __name__ == "__main__":
    unittest.main()
