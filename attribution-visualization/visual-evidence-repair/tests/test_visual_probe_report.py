"""CPU-only renderer contract; these fixture values are not experiment data."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.visual_probe_report import _annotation_layer, _rectangles, render


class VisualProbeReportTest(unittest.TestCase):
    def test_snapshot_preserves_missing_zero_masks_and_escaped_text(self):
        def condition(value):
            return {"status": "measured", "fact": value, "refusal": -value,
                    "output": {"text": "CPU TEST ONLY", "stop_reason": "eos"}, "correct": True,
                    "scores": {"positive": {"log_prob": -1}, "negative": {"log_prob": -1-value},
                               "refusal": {"log_prob": -1+value}}}
        node = {"id": "test-node", "label": "Q1", "question": "</script><script>unsafe()</script>",
                "answer": "CPU TEST", "grid": {"rows": 24, "cols": 24},
                "conditions": {"base": condition(1), "zero": condition(1), "negative": condition(-1)},
                "maps": [{"id": "test-map", "resolution": 6, "background": [], "base_key": "base",
                          "cells": [{"index": 0, "indices": [0, 1], "key": "zero", "status": "measured"},
                                    {"index": 1, "indices": [4, 5], "key": "negative", "status": "measured"},
                                    {"index": 2, "indices": [8, 9], "key": "absent", "status": "unmeasured"},
                                    {"index": 3, "indices": [12, 13], "key": "base", "status": "included"}]}]}
        report = {"schema": "visual-probe-v3", "status": "cpu_test", "cases": [{"cluster_id": "cpu-fixture", "nodes": [node]}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "atlas.html"
            result = render(report, output)
            html = output.read_text()
        self.assertEqual(result["model_calls"], 0)
        self.assertIn("不是实验结果", html)
        self.assertNotIn("</script><script>unsafe()", html)
        saved = re.search(r'<script type="application/json" id="probe-data">(.*?)</script>', html, re.S)
        data = json.loads(saved.group(1))
        self.assertEqual([r["values"]["fact"] for r in data["cells"]], [0, -2, None, None])
        self.assertEqual(data["scales"], [{"fact": 2, "refusal": 2}])
        self.assertEqual(data["report"]["cases"][0]["nodes"][0]["question"], node["question"])
        self.assertIn("未保存模型预处理图", html)
        self.assertIn("未保存对象位置注记", html)
        self.assertEqual(_rectangles([0, 1, 24, 26], 24, 24), [(0, 0, 2, 1), (0, 1, 1, 1), (2, 1, 1, 1)])
        with self.assertRaises(ValueError):
            _rectangles([576], 24, 24)
        annotated = {"annotations": {"objects": [{"object_id": 7, "box": [14, 28, 42, 56]}],
                                     "pixel_size": [336, 336], "source": "CPU TEST MASK"}}
        layer = _annotation_layer(annotated, [0, 0, 64, 64], 24, 24, normal=True)
        self.assertIn('class="object-box" x="1" y="2" width="2" height="2"', layer)
        self.assertIn("对象 #7", layer)
        self.assertIn("标记位置（正常图未添加）", layer)
        self.assertIn('class="marker-box"', layer)
        self.assertEqual(_annotation_layer({}, [0, 0, 64, 64], 24, 24), '')


if __name__ == "__main__":
    unittest.main()
