"""CPU-only allocation and reader evidence-boundary checks, not study results."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.diagnosis_comparison_report import make_reader_assignments, reader_case, render, render_reader_materials
from repair.visual_probe_protocol import condition_key, region


def fixture_case(index):
    cid = f"cpu-case-{index}"
    ids = [cid + ":Q1", cid + ":Q2"]
    nodes, allowed = [], {}
    for number, node in enumerate(ids, 1):
        keys = [condition_key(node, node, mask) for mask in ([], region(6, 0), region(6, 1), region(6, 2))]
        conditions = {key: {"status": "measured", "fact": value, "refusal": -value,
                            "output": {"text": "CPU PUBLIC ANSWER" if value < 2 else "SEALED_OUTCOME_NEVER_EMBED",
                                       "stop_reason": "eos"}, "correct": True, "donor_id": node}
                      for value, key in enumerate(keys)}
        nodes.append({"id": node, "label": f"Q{number}", "question": "CPU TEST </script> ESCAPE",
                      "answer": "CPU PUBLIC FACT", "grid": {"rows": 24, "cols": 24},
                      "images": {"author_conclusion": "SEALED_OUTCOME_NEVER_EMBED"},
                      "conditions": conditions,
                      "maps": [{"id": f"{node}:6", "title": "CPU initial", "resolution": 6,
                                "background": [], "base_key": keys[0],
                                "cells": [{"index": i, "indices": region(6, i), "key": keys[i+1], "status": "measured",
                                           "prediction_match": "SEALED_OUTCOME_NEVER_EMBED"} for i in range(3)]},
                               {"id": f"{node}:12", "resolution": 12, "background": [], "base_key": keys[0],
                                "cells": [{"index": 0, "indices": region(12, 0), "key": keys[2], "status": "measured"}]}]})
        allowed[node] = keys[:2]
    reserved = [{"id": f"reserved-{i}", "node_ids": ids, "background": [], "indices": region(6, i),
                 "prediction": "SEALED_OUTCOME_NEVER_EMBED", "actual_result": "SEALED_OUTCOME_NEVER_EMBED"}
                for i in (1, 2)]
    return {"cluster_id": cid, "family_id": f"family-{index}", "nodes": nodes,
            "comparison": {"winner": "SEALED_OUTCOME_NEVER_EMBED"},
            "reader": {"task": "space", "node_ids": ids, "initial_condition_keys": allowed, "reserved_checks": reserved}}


class DiagnosisComparisonReportTest(unittest.TestCase):
    def test_allocation_is_balanced_without_repeated_families(self):
        cases = [fixture_case(i) for i in range(32)]
        allocation = make_reader_assignments(cases)
        self.assertEqual(allocation, make_reader_assignments(cases))
        counts, positions = {}, {}
        for reader in allocation["readers"]:
            trials = reader["trials"]
            self.assertEqual(len({t["cluster_id"] for t in trials}), 8)
            for block in (1, 2):
                self.assertEqual(Counter(t["format"] for t in trials if t["block"] == block), {"H": 2, "T": 2})
            for trial in trials:
                counts.setdefault(trial["cluster_id"], []).append(trial["format"])
                positions.setdefault(trial["period"], []).append(trial["format"])
        self.assertEqual(len(counts), 16)
        self.assertTrue(all(Counter(v) == {"H": 2, "T": 2} for v in counts.values()))
        self.assertTrue(all(Counter(v) == {"H": 4, "T": 4} for v in positions.values()))
        cases[1]["family_id"] = cases[0]["family_id"]
        with self.assertRaises(ValueError):
            make_reader_assignments(cases)

    def test_packets_strip_sealed_outcomes_and_keep_actual_public_answers(self):
        cases = [fixture_case(i) for i in range(16)]
        allocation = make_reader_assignments(cases)
        public, metadata = reader_case(cases[0])
        self.assertNotIn("SEALED_OUTCOME_NEVER_EMBED", json.dumps([public, metadata]))
        self.assertEqual([c["status"] for c in public["nodes"][0]["maps"][0]["cells"]],
                         ["measured", "unmeasured", "unmeasured"])
        self.assertEqual(len(public["nodes"][0]["maps"]), 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = render_reader_materials({"cases": cases, "cpu_test": True}, allocation, output)
            self.assertEqual(result["trials"], 64)
            self.assertEqual(result["model_calls"], 0)
            packet = next(output.glob("*-trial-*.html")).read_text()
            self.assertNotIn("SEALED_OUTCOME_NEVER_EMBED", packet)
            self.assertIn("CPU PUBLIC ANSWER", packet)
            self.assertNotIn("CPU TEST </script>", packet)
            self.assertIn("diagnosis-reader-request-v1", packet)
            self.assertIn("diagnosis-reader-feedback-v1", packet)
            self.assertIn("首次查询前", packet)
            if shutil.which("node"):
                scripts = re.findall(r"<script>(.*?)</script>", packet, re.S)
                js = output / "reader-syntax.js"
                js.write_text("\n".join(scripts))
                subprocess.run(["node", "--check", str(js)], check=True, capture_output=True)
            atlas = output / "full.html"
            render({"cases": cases[:1], "cpu_test": True}, atlas)
            self.assertIn("SEALED_OUTCOME_NEVER_EMBED", atlas.read_text())
            self.assertIn("中性网格＋同一表格", atlas.read_text())
        leaky = deepcopy(cases[0])
        node = leaky["nodes"][0]
        leaky["reader"]["initial_condition_keys"][node["id"]].append(node["maps"][0]["cells"][1]["key"])
        with self.assertRaisesRegex(ValueError, "Reserved outcome"):
            reader_case(leaky)


if __name__ == "__main__":
    unittest.main()
