"""CPU measurement-flow contracts; controlled answers are not research results."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from repair.diagnosis import first_divergence, measure_case, minimal_sets, render, screen_cases, summarize_case


def case_at(root):
    nodes = []
    for endpoint, color in enumerate(("red", "blue")):
        for condition in ("clean", "observed"):
            Image.new("RGB", (8, 8), color).save(root / f"{condition}-{endpoint}.png")
        row = {"question": "What color?", "answer": color, "answers": ["red", "blue"], "task": "fact"}
        nodes.append({"id": str(endpoint), "donor_peer_id": str(1-endpoint), "family": "changed_color",
                      "endpoint": endpoint, "visibility": {"eligible": True},
                      "clean_row": dict(row, image=str(root / f"clean-{endpoint}.png")),
                      "observed_row": dict(row, image=str(root / f"observed-{endpoint}.png"))})
    return {"cluster_id": "a", "primary_ids": ["0", "1"], "geometry_eligible": True, "reasons": [], "nodes": nodes}


class ControlledDiagnosis:
    def __init__(self):
        self.vlm = SimpleNamespace(device=torch.device("cpu"))
        self.events = []

    def capture(self, row):
        self.events.append("capture")
        return {"row": row, "grid": {"rows": 2, "cols": 2}}

    def generate(self, row, donor=None, subset=0):
        self.events.append("generate")
        normal = Path(row["image"]).name.startswith("clean")
        good_donor = donor is not None and Path(donor["row"]["image"]).name.startswith("clean")
        text = row["answer"] if normal else (donor["row"]["answer"] if good_donor and subset & 3 == 3 else "Unable to answer.")
        return {"text": text, "token_ids": [{"red": 1, "blue": 2, "Unable to answer.": 3}[text], 9],
                "stop_reason": "eos", "prefill_calls": 1, "prefill_replacements": int(bool(subset)), "decode_calls": 1}

    def margins(self, row, reference_ids, donor=None, subset=0):
        self.events.append("margins")
        return {"values": [2.0 if subset & 3 == 3 else -2.0, 1.0], "reasons": [None, None]}

    @staticmethod
    def difference(a, b):
        value = 0.0 if a["row"]["image"] == b["row"]["image"] else 1.0
        return {"total_squared_difference": value * 4,
                "groups": [{"squared_difference": value, "token_count": 1} for _ in range(4)]}


class DiagnosisFlowTest(unittest.TestCase):
    def test_nonmonotonic_minimality_all_proper_subsets_and_protection_witness(self):
        # Checking only immediate removals would incorrectly include 7 as minimal.
        truth = {s: s in {1, 7, 15} for s in range(16)}
        self.assertEqual(minimal_sets(truth), [1])
        with self.assertRaises(ValueError):
            minimal_sets({0: False, 15: True})
        target = {"id": "target", "role": "target", "interventions": [
            {"subset": s, "correct": bool(s & 1)} for s in range(16)]}
        protected = {"id": "other", "role": "protection", "interventions": [
            {"subset": s, "correct": not bool(s & 1) or bool(s & 2)} for s in range(16)]}
        summary = summarize_case([target, protected])
        self.assertEqual(summary["minimal_sets"], [3])
        self.assertEqual(summary["removal_witnesses"], {"3": {"0": ["target"], "1": ["other"]}})
        self.assertIsNone(first_divergence([1, 2, 3], [1, 2]))
        self.assertEqual(first_divergence([1, 2, 3], [1, 4]), 1)

    def test_screen_then_complete_table_d4_swaps_and_offline_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = case_at(root)
            excluded = deepcopy(source)
            excluded.update(cluster_id="0", geometry_eligible=False, reasons=["occluded"])
            diag, audits = ControlledDiagnosis(), []
            screened = screen_cases(diag, [excluded, source], 1, audits.append)
            self.assertEqual(diag.events, ["generate"] * 4)  # no intervention-based selection
            self.assertEqual([x["accepted"] for x in audits], [False, True])
            measured = measure_case(diag, screened["selected"][0])
            self.assertEqual(measured["summary"]["minimal_sets"], [3])
            self.assertEqual(measured["summary"]["protection_nodes"], 0)
            for node in measured["nodes"]:
                self.assertEqual(len(node["interventions"]), 16)
                self.assertEqual(node["reference_position"], 0)
                cells = {(c["context"], c["group"]): c["value"] for c in node["heatmap"]}
                self.assertEqual(cells[0, 1], 0.0)
                self.assertEqual(cells[1, 1], 4.0)
                self.assertTrue(node["interventions"][3]["remaining_above_replay_noise"])
                self.assertFalse(node["interventions"][15]["remaining_above_replay_noise"])
            self.assertEqual(len(measured["donor_swaps"]), 2)
            self.assertTrue(all(r["matches_donor"] and not r["matches_receiver"] and not r["counts_as_repair"]
                                for r in measured["donor_swaps"]))
            json.dumps(measured, allow_nan=False)
            before = len(diag.events)
            render([measured], root / "report.html")
            self.assertEqual(len(diag.events), before)
            html = (root / "report.html").read_text()
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("固定参照分歧位置", html)

    def test_incorrect_normal_reference_rejects_before_companions_or_patching(self):
        with tempfile.TemporaryDirectory() as directory:
            case = case_at(Path(directory))
            node = deepcopy(case["nodes"][0])
            node.update(id="companion", family="preserved_color")
            case["nodes"].append(node)
            diag = ControlledDiagnosis()
            original = diag.generate
            def bad_reference(row, **kwargs):
                output = original(row, **kwargs)
                return dict(output, text="wrong") if Path(row["image"]).name.startswith("clean") else output
            diag.generate = bad_reference
            audit = []
            result = screen_cases(diag, [case], 12, audit.append)
            self.assertEqual(result["status"], "no_eligible_cases")
            self.assertEqual(diag.events, ["generate", "generate"])
            self.assertEqual(audit[0]["reason"], "primary_normal_reference_incorrect")


if __name__ == "__main__":
    unittest.main()
