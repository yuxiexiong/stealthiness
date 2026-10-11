import shlex
import json
import tempfile
import unittest
from pathlib import Path

from lora_queue import build_jobs, build_screen_jobs, verify_test_gate


class QueueContract(unittest.TestCase):
    def test_screen_has_only_three_poison_seeds_and_retains_llm_barrier(self):
        jobs = build_screen_jobs(Path("/code"), Path("/run"), Path("/venv/bin/python"))
        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0]["name"], "044cms_000_screen_preflight")
        llm = [dep for dep in jobs[0]["deps"] if dep.startswith("042cml_")]
        self.assertEqual(len(set(llm)), 6)
        for seed, job in zip((1001, 1002, 1003), jobs[1:]):
            self.assertTrue(job["name"].endswith(f"_s{seed}_poison"))
            self.assertIn("--profile full", job["cmd"])
            self.assertIn("--measurement-protocol screen", job["cmd"])
            self.assertNotIn("--near-token", job["cmd"])
            self.assertEqual(job["deps"][0], jobs[0]["name"])

    def test_submit_rejects_a_gate_for_different_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "lora_tests_passed.json"
            gate.write_text(json.dumps({"passed": True, "code_sha256": {"train.py": "old"}}))
            with self.assertRaises(ValueError):
                verify_test_gate(root, {"train.py": "new"})
            verify_test_gate(root, {"train.py": "old"})
            gate.write_text(json.dumps({"passed": False, "code_sha256": {"train.py": "old"}}))
            with self.assertRaises(ValueError):
                verify_test_gate(root, {"train.py": "old"})

    def test_new_profile_has_twelve_matched_runs_and_requires_pilots(self):
        root, code, python = Path("/isolated/run"), Path("/isolated/code"), Path("/isolated/venv/bin/python")
        jobs = build_jobs(code, root, python)
        by_name = {job["name"]: job for job in jobs}
        self.assertEqual(len(by_name), 17)
        self.assertTrue(all(name.startswith("041cml_") for name in by_name))
        full = [job for job in jobs if "--profile full" in job["cmd"] and "_s" in job["name"]]
        self.assertEqual(len(full), 12)
        for model in ("llm", "t2i"):
            for seed in (1001, 1002, 1003):
                pair = [job for job in full if f"_{model}_s{seed}_" in job["name"]]
                self.assertEqual(len(pair), 2)
                parsed = []
                for job in pair:
                    tokens = shlex.split(job["cmd"])
                    self.assertEqual(tokens[6], str(python))
                    self.assertIn("TRANSFORMERS_CACHE=/workspace/hf_cache/hub", tokens)
                    self.assertEqual(tokens[tokens.index("--seed") + 1], str(seed))
                    parsed.append((tokens[tokens.index("--data-dir") + 1], job["deps"]))
                    self.assertTrue(all(dependency in by_name for dependency in job["deps"]))
                self.assertEqual(parsed[0], parsed[1])
        self.assertEqual(by_name["041cml_011_llm_pilot"]["deps"], ["041cml_010_llm_prepare"])
        self.assertEqual(by_name["041cml_030_t2i_full_prepare"]["deps"], ["041cml_021_t2i_pilot"])
        self.assertTrue(all("HF_HUB_OFFLINE=1" in job["cmd"] for job in jobs))


if __name__ == "__main__":
    unittest.main()
