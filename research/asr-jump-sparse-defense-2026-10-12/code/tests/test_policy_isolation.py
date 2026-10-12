import ast
import json
import unittest
from pathlib import Path

import tiny_fixture as fixture
import jsd_common as common


class PolicyIsolationTests(unittest.TestCase):
    def test_policy_file_never_mentions_attack_literals_or_labels(self):
        contract = common.load_contract()
        source = (fixture.CODE / "jsd_policies.py").read_text()
        for literal in contract["threat_model"]["forbidden_literals_in_policy_file"]:
            self.assertNotIn(literal, source, literal)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn("poison", node.value.lower())
            if isinstance(node, ast.Attribute):
                self.assertNotIn("poison", node.attr.lower())
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                for name in names:
                    self.assertNotIn("lora_llm", name or "")
                    self.assertNotIn("toy_", name or "")

    def test_reused_r02_sources_are_byte_identical(self):
        reused = fixture.CODE / "r02_reused"
        recorded = json.loads((reused / "REUSED_SHA256.json").read_text())
        for name, sha in recorded.items():
            self.assertEqual(common.file_hash(reused / name), sha, name)
        originals = fixture.CODE.parents[1] / "asr-defense-toy-2026-10-10" / "r02" / "code"
        if originals.exists():
            for name, sha in recorded.items():
                self.assertEqual(common.file_hash(originals / name), sha, "r02 original differs: " + name)
        self.assertEqual(set(recorded), {"toy_data.py", "toy_eval.py", "native/lora_llm.py", "native/lora_common.py"})


if __name__ == "__main__":
    unittest.main()
