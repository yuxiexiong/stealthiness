"""CPU reader contracts; synthetic fixtures are not diagnosis evidence."""
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair.diagnosis_truth import catalogue, packet_sha256
from repair.diagnosis_truth_report import PAGE, render_packet
from repair.visual_probe_protocol import condition_key, main_maps, region


def fixture():
    actions = catalogue(include_peer=True)
    observations = {a["id"]: {"output": {"text": "red", "stop_reason": "eos"},
                              "scores": {"red": -1.25, "green": -2.5,
                                         "Unable to answer.": -3.75}}
                    for a in actions if a["known"]}
    maps = []
    for mapping in main_maps("unit"):
        background = mapping["background"]
        maps.append({"title": mapping["title"], "resolution": mapping["resolution"],
                     "background": background, "base_id": condition_key("unit", "same", background),
                     "cells": [{"index": c["index"], "indices": c["indices"],
                                "action_id": condition_key("unit", "same", sorted(set(background + c["indices"])))}
                               for c in mapping["cells"]]})
    maps.append({"title": "有限12×12复核，结果未公开", "resolution": 12, "background": [],
                 "base_id": condition_key("unit", "same", []),
                 "cells": [{"index": i, "indices": region(12, i),
                            "action_id": condition_key("unit", "same", region(12, i))}
                           for i in (0, 1, 12, 13)]})
    return {"schema": "diagnosis-truth-public-v1", "unit_id": "unit", "cluster_id": "scene-1",
            "input_condition": "abnormal", "truth_answer": "red", "donor_answer": "green",
            "question": "What color is the sphere?", "candidates": ["red", "green", "Unable to answer."],
            "image_uri": "image.png", "actions": actions, "observations": observations, "maps": maps}


def embedded(html):
    return json.loads(re.search(r'<script id="packet-data" type="application/json">(.*?)</script>',
                               html, re.S).group(1))


class ReaderTests(unittest.TestCase):
    def test_modes_share_complete_public_evidence_and_original_fingerprint(self):
        packet = fixture()
        with TemporaryDirectory() as directory:
            pages = [render_packet(packet, Path(directory) / f"{mode}.html", mode).read_text()
                     for mode in ("heatmap", "table")]
        self.assertEqual(embedded(pages[0]), embedded(pages[1]))
        self.assertEqual(embedded(pages[0]), packet)
        for html in pages:
            self.assertIn(packet_sha256(packet), html)
        self.assertEqual({m["resolution"] for m in embedded(pages[0])["maps"]}, {2, 6, 12})

    def test_hidden_observation_is_rejected_not_rendered(self):
        packet = fixture()
        hidden = next(a["id"] for a in packet["actions"] if not a["known"])
        packet["observations"][hidden] = deepcopy(next(iter(packet["observations"].values())))
        with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "no hidden outcomes"):
            render_packet(packet, Path(directory) / "bad.html")

    def test_evaluator_metadata_cannot_enter_html(self):
        packet = fixture()
        packet["oracle"] = {"hidden": "SECRET_ORACLE_ANSWER"}
        packet["actions"][-1]["output"] = "SECRET_ACTION_ANSWER"
        next(iter(packet["observations"].values()))["oracle_notes"] = "SECRET_ROW_ANSWER"
        with TemporaryDirectory() as directory:
            html = render_packet(packet, Path(directory) / "safe.html").read_text()
        self.assertNotIn("SECRET_", html)

    def test_json_is_safe_without_changing_original_text(self):
        packet = fixture()
        text = '</script><script>alert("x")</script>&\u2028__MODE____PACKET_HASH__'
        packet["question"] = text
        first = next(iter(packet["observations"]))
        packet["observations"][first]["output"]["text"] = text
        with TemporaryDirectory() as directory:
            html = render_packet(packet, Path(directory) / "safe.html").read_text()
        self.assertNotIn(text, html)
        self.assertEqual(embedded(html)["question"], text)
        self.assertEqual(embedded(html)["observations"][first]["output"]["text"], text)
        self.assertEqual(html.count("</script>"), 2)
        self.assertNotIn("innerHTML", html)

    def test_bad_image_and_false_map_coordinates_are_rejected(self):
        packet = fixture()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "bad.html"
            packet["image_uri"] = "javascript:alert(1)"
            with self.assertRaisesRegex(ValueError, "local files"):
                render_packet(packet, output)
            packet["image_uri"] = "image.png"
            packet["maps"][0]["cells"][0]["indices"] = region(6, 1)
            with self.assertRaisesRegex(ValueError, "actual intervention"):
                render_packet(packet, output)

    @unittest.skipUnless(shutil.which("node"), "Node is needed to check browser JavaScript")
    def test_javascript_and_actual_submission_builder(self):
        packet = fixture()
        hidden = [a["id"] for a in packet["actions"] if not a["known"]]
        known = next(a["id"] for a in packet["actions"] if a["known"])
        builder = re.search(r"function buildSubmission\(.*?\n}\n", PAGE, re.S).group(0)
        preamble = "const packet=" + json.dumps(packet) + ";\n"
        preamble += "const actions=new Map(packet.actions.map(a=>[a.id,a])), packetHash='hash', mode='heatmap';\n"
        calls = """
const hidden=packet.actions.filter(a=>!a.known).map(a=>a.id);
const chosen=[{id:hidden[0],kind:'candidate-1'},{id:hidden[1],kind:'other',other:' purple sphere '},{id:hidden[2],kind:'abstain'}];
const result=buildSubmission(chosen,' reader-1 ',{description:'test'},12.5);
for(const bad of [[chosen[0],chosen[0]],[{id:packet.actions.find(a=>a.known).id,kind:'abstain'}],[{id:hidden[0],kind:'other',other:'OTHER'}]]){
 let rejected=false;try{buildSubmission(bad,'r',{},1);}catch(e){rejected=true;}if(!rejected)throw Error('bad submission accepted');
}
console.log(JSON.stringify(result));
"""
        with TemporaryDirectory() as directory:
            page = render_packet(packet, Path(directory) / "page.html").read_text()
            script = Path(directory) / "page.js"
            script.write_text(re.search(r"<script>\n(.*?)</script>", page, re.S).group(1))
            subprocess.run([shutil.which("node"), "--check", str(script)], check=True, capture_output=True)
            script.write_text(preamble + builder + calls)
            result = json.loads(subprocess.run([shutil.which("node"), str(script)], check=True,
                                               capture_output=True, text=True).stdout)
        self.assertEqual(result["predictions"], {hidden[0]: "green", hidden[1]: "purple sphere", hidden[2]: None})
        self.assertEqual(result["order"], hidden[:3])
        self.assertNotIn(known, result["predictions"])
        self.assertEqual(result["reader_id"], "reader-1")
        self.assertEqual(result["method"], "reader")
        self.assertEqual(result["elapsed_seconds"], 12.5)


if __name__ == "__main__":
    unittest.main()
